import os
from models.TransVW.ynet3d import *  # TransVW
from models.ModelsGenesis.unet3d import UNet3D as ModelsGenesis  # ModelsGenesis
from monai.networks.nets.swin_unetr import SwinUNETR # SwinUNETR
from models.GVSL.UNet_GVSL import UNet3D_GVSL  # GVSL
from models.PCRLv2 import PCRLv23d
from models.UniMiSS.MiT import MiTnet
from models.vox2vec.nn import FPN3d, FPNLinearHead
from models.vox2vec.eval.end_to_end import EndToEnd
from models.UNETR.unetr import UNETR
from models.HySparK.hybird3 import build_hybird
from models.SparK.mednext_spark import MedNeXt

def load_TransVW(num_classes):
    model = UNet3D(n_class=num_classes)
    weight_dir = './ckpt/TransVW_chest_ct.pt'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    delete = [key for key in state_dict if "projection_head" in key]
    for key in delete: del state_dict[key]
    delete = [key for key in state_dict if "prototypes" in key]
    for key in delete: del state_dict[key]
    for key in state_dict.keys():
        if key in model.state_dict().keys():
            model.state_dict()[key].copy_(state_dict[key])
            print("Copying {} <---- {}".format(key, key))
        elif key.replace("classficationNet.", "") in model.state_dict().keys():
            model.state_dict()[key.replace("classficationNet.", "")].copy_(state_dict[key])
            print("Copying {} <---- {}".format(key.replace("classficationNet.", ""), key))
        else:
            print("Key {} is not found".format(key))
    model.cuda()
    return model

def load_ModelsGenesis(num_classes):
    model = ModelsGenesis(n_class=num_classes)
    weight_dir = './ckpt/Genesis_Chest_CT.pt'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        if "out_tr.final_conv" not in key:
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    print(unParalled_state_dict.keys())
    model.load_state_dict(unParalled_state_dict, strict=False)
    model.cuda()
    return model


def load_SwinUNETR(num_classes):
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=num_classes,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    )
    model_dict = torch.load("./ckpt/model_swinvit.pt")
    state_dict = model_dict["state_dict"]
    # fix potential differences in state dict keys from pre-training to
    # fine-tuning
    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
    # We now load model weights, setting param `strict` to False, i.e.:
    # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
    # the decoder weights untouched (CNN UNet decoder).
    model.load_state_dict(state_dict, strict=False)
    print("Using pretrained self-supervised Swin UNETR backbone weights !")
    model.cuda()
    return model

def load_GVSL(num_classes):
    model = UNet3D_GVSL(n_classes=num_classes, pretrained_dir='./ckpt/GVSL_epoch_1000.pth').cuda()
    return model


def load_prlv2(num_classes):
    # pip install segmentation-models-pytorch
    model = PCRLv23d(n_class=num_classes)
    weight_path = './ckpt/simance_multi_crop_luna_pretask_1.0_240.pt'
    model_dict = torch.load(weight_path)['state_dict']
    model_dict.pop("out_tr.final_conv.weight")
    model_dict.pop("out_tr.final_conv.bias")
    model.load_state_dict(model_dict, strict=False)
    model.cuda()
    return model

def load_UniMiss(num_classes):
    model = MiTnet(pretrain=True, pretrain_path="./ckpt/UniMiss_small.pth", activation_cfg='LeakyReLU',
           norm_cfg='IN', img_size=[96, 96, 96], num_classes=num_classes)
    model.cuda()
    return model

def load_VoCo(num_classes):
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=num_classes,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
        use_v2=True)
    model_dict = torch.load("./ckpt/VoCo_10k.pt", map_location=torch.device('cpu'))
    state_dict = model_dict
    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print("Using pretrained voco ema self-supervised Swin UNETR backbone weights !")
    model.cuda()
    return model


def load_vox2vec(num_classes):
    backbone = FPN3d(1, 16, 6)
    flag = backbone.load_state_dict(torch.load("./ckpt/vox2vec.pt"))
    head = FPNLinearHead(16, 6, num_classes)
    model = EndToEnd(backbone, head, patch_size=(96, 96, 96)).cuda()
    if flag:
        print("Using pretrained vox2vec weights !")
    return model

def load_spark(num_classes):
    model = MedNeXt(
        in_channels=1,
        n_classes=num_classes,
        n_channels=32,
        exp_r=[2,3,4,4,4,4,4,3,2],
        kernel_size=3,
        do_res=True,
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2],
        sparse=False
    )
    model_dict = torch.load("./ckpt/spark.pth")
    flag = model.load_state_dict(model_dict, strict=False)  
    if flag:
        print("SparK use pretrained weights Successful")
    model.cuda()
    return model

def load_hyspark(num_classes):
    model = build_hybird(in_channel=1, n_classes=num_classes, img_size=96).cuda()
    model_dict = torch.load("./ckpt/hyspark.pth")
    flag = model.load_state_dict(model_dict, strict=False)  
    if flag:
        print("SparK use pretrained weights Successful")
    model.cuda()
    return model
 

def load_hi_end_mae_10k(num_classes):
    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=(96, 96, 96),
        feature_size=32,
        hidden_size=1536,
        mlp_dim=1536 * 4,
        num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        qkv_bias=True
        )
    model_dict = torch.load("./ckpt/hi_end_mae_10k.pth")['state_dict']
    pretrained_dict = {}
    pretrained_dict["vit.patch_embedding.patch_embeddings.1.weight"] = model_dict["module.encoder.patch_embed.proj.weight"].flatten(1)
    pretrained_dict["vit.patch_embedding.patch_embeddings.1.bias"] = model_dict["module.encoder.patch_embed.proj.bias"]
    pretrained_dict["vit.patch_embedding.position_embeddings"] = model_dict["module.encoder_pos_embed"]
    for k, v in model_dict.items():
        if "encoder." in k and "cls_token" not in k and "patch_embed" not in k:
            pretrained_dict[k.replace("module.encoder", "vit").replace("fc", "linear").replace("proj", "out_proj")] = v
    model.load_state_dict(pretrained_dict, strict=False)
    model.cuda()
    return model





