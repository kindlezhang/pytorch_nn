import torch
import torch.nn as nn
from functools import partial

class PatchEmbed(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_c = 3, embed_dim = 768, norm_layer = None):
        # img_size:图像大小 patch size每个patch的大小 每行14个patch
        super().__init__()
        img_size = (img_size, img_size) # 将输入的图像大小变为二维元组
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1]) # patch的网格大小 224//16 =14 （14,14）
        self.num_patches = self.grid_size[0]*self.grid_size[1] # 14*14=196 patch的总数

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size = patch_size, stride = patch_size) # 3, 224, 224 -> 768, 14, 14 
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # 若有layer norm则使用 若无则保持不变

    def forward(self, x):
        B, C, H, W = x.shape # 获取我们输入张量的形状
        assert H == self.img_size[0] and W==self.img_size[1],\
        f"输入图像的大小{H}*{W}与模型期望大小{self.img_size[0]}*{self.img_size[1]}不匹配"
        # B, 3, 224, 224 -> B, 768, 14, 14 -> B, 768, 196 -> B, 196, 768
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x) # 若有归一化层则使用
        return x

class Attention(nn.Module):
    def __init__(self, 
                dim, # 输入的token维度, 768
                num_heads = 8, # 注意力头数
                qkv_bias = False, # 生成QKV的时候是否增加偏置
                qk_scale = None, # 用于缩放QK的系数， 如果是None, 则使用1/sqrt(embed_dim_pre_head)
                atte_drop_ration = 0., # 注意力分数的dropout的比率，防止过拟合
                proj_drop_ration = 0.): # 最终投影层的dropout比例
        super().__init__()
        self.num_heads = num_heads 
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** 0.5 # qk的缩放因子
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias) # 通过全连接层生成QKV，为了并行计算，提高计算效率，参数更少
        self.att_drop = nn.Dropout(atte_drop_ration)
        self.proj_drop = nn.Dropout(proj_drop_ration)
        # 将每个head得到的输出进行concat拼接，然后通过线性变换映射回原本的嵌入的dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape # batch, num, num_patchs + 1, embed_dim 这个1为class token
        # B N 3*C -> B N 3 num_heads C//self.num_heads -> 3, B, number_heads, N, C//self.num_heads
        qkv = self.qkv(x).reshape(B,N, 3, self.number_heads, C//self.number_heads).permute(2,0,3,1,4) # 方便之后我们做运算
        # 用切片的方法拿到QKV
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 计算qk的点积，并进行缩放 得到注意力分数
        # k.transpose(-2, -1) 
        attn = (q @ k.transpose(-2, -1))*self.scale # B, num_heads, N, N
        atten = atten.softmax(dim = -1) # 在最后一个维度（行）上进行处理，每行总和为1
        # v: B, number_heads, N, C//self.num_heads
        # atten @ v: B, number_heads, N, C//self.num_heads -> B, N, number_heads, C//self.num_heads -> B N C
        x = (atten @ v).transpose(1,2).reshape(B,N,C) 
        # 通过线性变换映射回原本嵌入的dim
        x = self.proj(x)
        x = self.proj_drop(x) # 防止过拟合

        return x 
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features=None, act_layer = nn.GELU,drop=0.):
        # in_features输入的维度，hidden_features隐藏层的维度通常为in_features的4倍, out_features通常与输入维度相等
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x) # 第一个全连接层
        x = self.act(x) # 激活函数
        x = self.drop(x) # 丢弃一定神经元
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self,
                 dim, # 每个token的维度 768
                 num_heads, # 头数
                 mlp_ratio=4, # 计算hidden_features大小 为输入的4倍
                 qkv_bias = False,
                 qk_scale = None,
                 drop_ratio=0., # 多头自注意力机制最后的linear后使用的dropout
                 attn_drop_ratio = 0., # 生成qkv之后的dropout
                 drop_path_ratio = 0., # drop_path的比例
                 act_layer = nn.GELU, # 对应激活函数
                 norm_layer = nn.LayerNorm): # 正则化层
        super(Block,self).__init__()
        self.norm1 = norm_layer(dim) # transformer encoder block中的第一个layer norm
        self.attn = Attention(dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale,
                              atte_drop_ration=attn_drop_ratio, proj_drop_ration=drop_ratio)
        # 如果 drop_path_ratio>0，则使用droppath,否则不做任何更改
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio>0. else nn.Identity()
        self.norm2 = norm_layer(dim) # 定义第二个layer_norm层
        mlp_hidden_dim = int(dim*mlp_ratio) # 计算mlp第一个全连接层的节点个数
        # 定义mlp层 传入dim=mlp_hidden dim
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer = act_layer, drop = drop_ratio)

    def forward(self,x):
        x = x + self.drop_path(self.attn(self.norm1(x))) # 前向传播
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_c = 3, num_classes = 10000,
                 embed_dim=768, depth=12, num_heads = 12, mlp_ratio = 4.0, qkv_bias = True, 
                 qk_scale = None, representation_size = None, distilled = False, drop_ratio = 0.,
                 attn_drop_ratio = 0., drop_path_ratio = 0., embed_layer = PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer,self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim # embed_dim 赋值给self.embed_dim和self.num_features
        self.num_tokens = 2 if distilled else 1 # numtokens 为1
        # 设置一个较小的参数防止分母为0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps = 1e-6)
        act_layer = act_layer or nn.GELU()
        self.path_embed = embed_layer(img_size=img_size, patch_size = patch_size, in_c = in_c, embed_dim = embed_dim)
        num_patches = self.path_embed.num_patches # 得到patches的个数
        # 使用nn.Parameter构建可训练的参数，用零矩阵进行初始化，第一个为batch的维度
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1,1,embed_dim)) if distilled else None
        # pos_embed大小与concat拼接之后的大小一致，197， 768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p = drop_ratio)
        # 根据传入的drop_path_ratio构建等差序列从0到drop_path_ratio, 有depth个元素
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # 使用nn.Sequential将列表中的所有模块打包为一个列表
        self.block = nn.Sequential(*[
            Block(dim = embed_dim, num_heads =num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio = drop_ratio, attn_drop_ratio = attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer = norm_layer,act_layer = act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) # 通过transformer后的layernorm

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrdereDict([
                ("fc", nn.Linear(embed_dim, representation_size)), 
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity() # pre_logits不做任何处理
        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes>0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes>0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std = 0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token,std=0.02)

        nn.init.trunc_normal_(self.cls_token, std = 0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # B C H W -> b num_patches embed_dim
        x = self.patch_embed(x)
        # 1,1,768 -> B, 1, 768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 如果dist_token存在则拼接dist_token和cls_token,否则只拼接cls_token和输入的patch特征x
        if self.dist_token is None:
            x = torch.cat((cls_token,x), dim = 1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1),x),dim=1)
        
        x = self.pos_drop(x + self.pos_embed)
        x = self.block(x)
        x = self.norm(x)
        if self.dist_token is None: # dist_token为None 提取cls_token对应的输出 
            return self.pre_logits(x[:,0])
        else:
            return x[:,0],x[:,1]
        
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            # 如果是训练模式且不是脚本模式
            if self.training and not torch.jit.is_scripting():
                # 则返回两个头部的结果
                return x, x_dist
        else:
            x = self.head(x) # 最后的linear全连接层
        return x
    
def _init_vit_weights(m):
    # 判断模块m是否是nn.Linear
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None: # 如果线性层存在偏置项
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode = "fan_out") # 对卷积层的权重做一个初始化 适用于卷积
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight) # 对层归一化的权重初始化为1

def vit_base_patch16_224(num_classes:int = 1000, pretrained=False):
    model = VisionTransformer(img_size = 224,
                              patch_size= 16,
                              embed_dim = 768,
                              depth  = 12,
                              num_heads = 12,
                              representation_size = None,
                              num_classes =num_classes)
    return model