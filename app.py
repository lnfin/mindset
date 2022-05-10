import os
import streamlit as st
import numpy as np
from PIL import Image
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from modeling.models import MODNet, Unet
from utils.cfgtools import Config

# https://drive.google.com/uc?id=18XL-UDLxMXjCQz751xYsoXPXYr-q1L-0&export=download&confirm=t
# https://drive.google.com/uc?id=17SahTKf12vbtu5Yb2qrcXbj09lKsGHK_&export=download&confirm=t
drive_links = {
    "unet.pth": "https://drive.google.com/uc?id=17SahTKf12vbtu5Yb2qrcXbj09lKsGHK_&export=download&confirm=t",
    "modnet.ckpt": "https://drive.google.com/uc?id=1oqMQK314qgsBEO6fFqFAqVYfLT2Kz1gr&authuser=0&export=download",
    "test.cfg": "https://drive.google.com/uc?id=17gXaQrl3nsH6Z5uZbsJ11K8sqODPBx56&authuser=0&export=download"
}

@st.cache
def download_model():
    for name, path in drive_links.items():
        gdown.download(path, name, quiet=False, use_cookies=False)


@st.cache
def cached_get_modnet(path_to_weights):
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    weights = torch.load(path_to_weights,
                         map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet


@st.cache
def cached_get_unet(path_to_config, path_to_weights):
    config = Config()
    unet = Unet(config.load(path_to_config))

    weights = torch.load(path_to_weights,
                         map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    unet.load_state_dict(weights)
    unet.eval()
    return unet


def main():
    download_model()
    modnet_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    unet_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512, 512))
        ]
    )
    ref_size = 512

    modnet = cached_get_modnet('modnet.ckpt')
    unet = cached_get_unet('test.cfg', 'unet.pth')

    if not os.path.exists('input/'):
        os.mkdir('input/')

    st.title('Удаление фона с изображения')
    st.subheader("Загрузка файлов")
    files = st.file_uploader('Выберите или ператащите сюда снимки', type=['png', 'jpeg', 'jpg'],
                                 accept_multiple_files=True)
    model_type = st.radio('Тип фото', ['Modnet', 'Unet'])

    if st.button('Загрузить') and files:
        results = {}
        for file in files:
            path = os.path.join('input', file.name)
            with open(path, 'wb') as f:
                f.write(file.getvalue())
            orig_img = Image.open(path)

            if model_type == 'Modnet':
                im = np.asarray(orig_img)
                if len(im.shape) == 2:
                    im = im[:, :, None]
                if im.shape[2] == 1:
                    im = np.repeat(im, 3, axis=2)
                elif im.shape[2] == 4:
                    im = im[:, :, 0:3]

                im = Image.fromarray(im)
                im = modnet_transform(im)

                im = im[None, :, :, :]

                im_b, im_c, im_h, im_w = im.shape
                if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                    if im_w >= im_h:
                        im_rh = ref_size
                        im_rw = int(im_w / im_h * ref_size)
                    elif im_w < im_h:
                        im_rw = ref_size
                        im_rh = int(im_h / im_w * ref_size)
                else:
                    im_rh = im_h
                    im_rw = im_w

                im_rw = im_rw - im_rw % 32
                im_rh = im_rh - im_rh % 32
                im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

                _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

                matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
                matte = matte[0][0].data.cpu().numpy() > 0.5
                matte_name = file.name.split('.')[0] + '.png'
                matte = orig_img * np.stack([matte, matte, matte]).transpose((1, 2, 0))

            else:
                im = unet_transform(orig_img)
                pred = unet(torch.unsqueeze(im, dim=0))
                inv_t = transforms.Resize((np.array(orig_img).shape[:2]))
                pred = np.array(inv_t(Image.fromarray(pred[0][0].detach().numpy()))) > 0.5
                matte = orig_img * np.stack([pred, pred, pred]).transpose((1, 2, 0)).astype(np.uint8)

            results[file.name] = matte.astype('uint8')

        with st.expander("Результат работы"):
            for name, image in results.items():
                st.markdown(f'<h3>{name}</h3>', unsafe_allow_html=True)
                st.image(image, width=350)

if __name__ == '__main__':
    main()

