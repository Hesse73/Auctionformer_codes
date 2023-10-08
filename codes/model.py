import torch
import torch.nn as nn
from torchvision.ops import MLP
import transformers
import math


def select_model(configs):
    if configs.model_name == 'bert':
        if configs.from_pretrain:
            bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        else:
            bert = transformers.BertModel(config=transformers.BertConfig())
        bert.encoder.layer = bert.encoder.layer[:configs.n_layers]
        d_model = 768

    elif configs.model_name =='bert_small':
        # default use pretrain
        bert = transformers.AutoModel.from_pretrained("prajjwal1/bert-small")
        d_model = 512
    elif configs.model_name == 'bert_large':
        bert = transformers.BertModel.from_pretrained("bert-large-uncased")
        d_model = 1024
    elif configs.model_name =='gpt-2':
        bert = transformers.GPT2Model.from_pretrained('gpt2')
        d_model = 768

    return bert, d_model


class BertValuedSolver(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # if configs.model_name =='bert':
        #     if configs.from_pretrain:
        #         self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        #     else:
        #         self.bert = transformers.BertModel(config=transformers.BertConfig())
        #     self.bert.encoder.layer = self.bert.encoder.layer[:configs.n_layers]

        self.bert, configs.d_model = select_model(configs)

        for name, param in self.bert.named_parameters():
            if configs.pretrain_requires_grad:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.mechanism_embedding = nn.Embedding(4, configs.d_model)
        if configs.entry_emb:
            print('using preset entry embedding size')
            self.entrance_embedding = nn.Embedding(configs.entry_emb, configs.d_model)
        else:
            self.entrance_embedding = nn.Embedding(configs.max_entry + 1, configs.d_model)
        self.value_dist_projection = nn.Linear(configs.valuation_range, configs.d_model)

        self.query_style = configs.query_style
        self.detach_branch = configs.detach_branch

        attn_dim = configs.d_model
        if self.query_style == 'branch_rcat':
            attn_dim = configs.d_model * 2
        self.decoder_k = nn.Linear(configs.d_model, attn_dim)
        self.decoder_v = nn.Linear(configs.d_model, attn_dim)

        if self.query_style == 'test_raw':
            query_input_dim = configs.valuation_range
        elif self.query_style == 'cat':
            query_input_dim = configs.valuation_range * 2
        elif self.query_style == 'branch_lcat':
            query_input_dim = configs.valuation_range + configs.d_model
        else:
            query_input_dim = configs.valuation_range

        if self.query_style == 'branch_add':
            # self.q_norm = nn.BatchNorm1d(configs.d_model)
            # self.vh_norm = nn.BatchNorm1d(configs.d_model)
            self.q_norm = nn.LayerNorm(configs.d_model)

        if configs.query_mlp:
            hidden_layers = [configs.hidden_dim] * configs.mlp_layers + [configs.d_model]
            self.query_projection = MLP(in_channels=query_input_dim, hidden_channels=hidden_layers)
        else:
            self.query_projection = nn.Linear(query_input_dim, configs.d_model)

        if configs.use_pos_emb:
            max_len = configs.valuation_range * configs.max_player
            if max_len >= 1024:
                self.position_embedding = LearnablePositionalEncoding(attn_dim, max_len=max_len+100)
            else:
                self.position_embedding = LearnablePositionalEncoding(attn_dim, max_len=1024)
        else:
            self.position_embedding = None

        self.attention = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=16, batch_first=True)

        hidden_layers = [configs.hidden_dim] * configs.mlp_layers + [configs.valuation_range]  # last_dim is V
        if configs.final_mlp:
            self.bid_mlp = MLP(in_channels=attn_dim, hidden_channels=hidden_layers)
        else:
            self.bid_mlp = nn.Linear(attn_dim, configs.valuation_range)

    def forward(self, x):
        # input: (mechanism:B, entry:B, value_dists:B*N*V, player_values:B*(VN)*V)
        mechanism, entry, value_dists, player_values = x
        B, N, V = value_dists.shape

        mechanism_emb = self.mechanism_embedding(mechanism)  # B*emb
        entry_emb = self.entrance_embedding(entry)  # B*emb
        value_dists_emb = self.value_dist_projection(value_dists)  # B*N*emb
        # B*(2+N)*emb
        input_embedding = torch.cat((mechanism_emb.unsqueeze(1), entry_emb.unsqueeze(1),
                                     value_dists_emb), dim=1)
        outputs = self.bert(inputs_embeds=input_embedding).last_hidden_state

        # bidding distribution
        k, v = self.decoder_k(outputs), self.decoder_v(outputs)  # B*(2+N)*attn_emb

        if self.query_style == 'test_raw':
            q = self.query_projection(player_values)  # B*VN*emb (no value_dist added, only for testing)
        elif self.query_style == 'cat':
            # cat values with repeated value_hist
            value_query = torch.cat((value_dists.repeat(1,V,1),player_values), dim=-1)  # B*VN*2V
            q = self.query_projection(value_query)  # B*VN*emb
        elif self.query_style == 'branch_lcat':
            # cat values with value_hist embedding
            branched_vh = value_dists_emb.detach() if self.detach_branch else value_dists_emb  # B*N*emb
            value_query = torch.cat((branched_vh.repeat(1,V,1), player_values), dim=-1)  # B*VN*(emb+V)
            q = self.query_projection(value_query)  # B*VN*emb
        elif self.query_style == 'branch_rcat':
            # cat value_hist embedding to q
            branched_vh = value_dists_emb.detach() if self.detach_branch else value_dists_emb  # B*N*emb
            q = self.query_projection(player_values)  # B*VN*emb
            q = torch.cat((branched_vh.repeat(1,V,1), q), dim=-1)  # B*VN*(2emb)
        else:
            # add value_hist embedding to q
            # branched_vh = value_dists_emb.detach() if self.detach_branch else value_dists_emb  # B*N*emb
            # q = self.query_projection(player_values)  # B*VN*emb
            # q = self.q_norm(q.transpose(1,2)).transpose(1,2)  # batch norm(B,emb,VN) --> B*VN*emb
            # branched_vh = self.vh_norm(branched_vh.repeat(1,V,1).transpose(1,2)).transpose(1,2)  # B*VN*emb
            # q = q + branched_vh  # B*VN*emb
            q = self.query_projection(player_values)  # B*VN*emb
            branched_vh = value_dists_emb.detach() if self.detach_branch else value_dists_emb
            q = self.q_norm(q + branched_vh.repeat(1,V,1))

        if self.position_embedding is not None:
            q = self.position_embedding(q)  # position embedding

        attn_output, _ = self.attention(q, k, v)  # B*VN*emb
        bid_dists = self.bid_mlp(attn_output)  # B*VN*V
        bid_dists = bid_dists.view(B,V,N,V)  # B*V*N*V
        bid_dists = bid_dists.transpose(1,2)  # B*N*V*V

        return bid_dists


class BertSolver(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.bert.encoder.layer = self.bert.encoder.layer[:configs.n_layers]

        for name, param in self.bert.named_parameters():
            if configs.pretrain_requires_grad:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.mechanism_embedding = nn.Embedding(4, configs.d_model)
        if configs.entry_emb:
            print('using preset entry embedding size')
            self.entrance_embedding = nn.Embedding(configs.entry_emb, configs.d_model)
        else:
            self.entrance_embedding = nn.Embedding(configs.max_entry + 1, configs.d_model)

        self.value_dist_projection = nn.Linear(configs.valuation_range, configs.d_model)
        output_dim = configs.valuation_range * configs.valuation_range
        # self.token_decoder = nn.Linear(768, output_dim)
        hidden_layers = [configs.hidden_dim] * configs.mlp_layers + [output_dim]  # last_dim is V*V
        if configs.final_mlp:
            print('using MLP as bidding predict')
            self.bid_mlp = MLP(in_channels=configs.d_model, hidden_channels=hidden_layers)
        else:
            self.bid_mlp = nn.Linear(configs.d_model, output_dim)

    def forward(self, x):
        # input: (mechanism:B, entry:B, value_dists:B*N*V)
        mechanism, entry, value_dists = x
        B, N, V = value_dists.shape

        mechanism = self.mechanism_embedding(mechanism)  # B*emb
        entry = self.entrance_embedding(entry)  # B*emb
        value_dists = self.value_dist_projection(value_dists)  # B*N*emb
        # B*(2+N)*emb
        input_embedding = torch.cat((mechanism.unsqueeze(1), entry.unsqueeze(1), value_dists), dim=1)

        outputs = self.bert(inputs_embeds=input_embedding).last_hidden_state
        y = self.bid_mlp(outputs)  # B*(2+N)*(V*V)
        y = y[:, -N:, :]  # B*N*(V*V)
        y = y.view(B, N, V, V)  # B*N*V*V
        y = y.softmax(dim=-1)  # softmax over last dim (each bid's probability)
        return y


class MLPSolver(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.max_player = configs.max_player
        self.valuation_range = configs.valuation_range
        self.max_entry = configs.max_entry

        self.emb_size = 128
        self.mechanism_embedding = nn.Embedding(4, self.emb_size)
        self.entrance_embedding = nn.Embedding(self.max_entry + 1, self.emb_size)

        input_size = self.max_player * self.valuation_range + self.emb_size * 2
        output_size = self.max_player * self.valuation_range * self.valuation_range

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        # input: (mechanism:B, entry:B, value_dists:B*N*V)
        mechanism, entry, value_dists = x

        mechanism = self.mechanism_embedding(mechanism)  # B*emb
        entry = self.entrance_embedding(entry)  # B*emb
        value_dists = value_dists.flatten(start_dim=1)  # B*(N*V)
        inputs = torch.cat((mechanism, entry, value_dists), dim=-1)
        y = self.model(inputs)  # B*(N*V*V)
        y = y.view(y.shape[0], self.max_player, self.valuation_range, self.valuation_range)  # B*N*V*V
        y = y.softmax(dim=-1)

        return y


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[0, x.size(1), :] * 1e-3


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1,max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.001, 0.001)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size,sequence length， embed dim]
            output: [ batch size, sequence length,embed dim]
        """

        x = x + self.pe[0,x.size(1), :]#*1e-3
        return self.dropout(x)