import os
import time

from PIL import Image
from torchvision import transforms
from model import *
from pathlib import Path
import utils
import numpy as np


def test_only_product(test_root, device, encoder, decoder, echo=None, steps=1, image_size=None, num_store=None, save_dir="./outputs"):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    if image_size is not None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # content_dir = Path(test_root)
    # content_paths = sorted([f for f in content_dir.glob('*')])
    content_paths = sorted(os.listdir(test_root))

    count = 1
    times = []
    for content_path in content_paths:
        print('product count:', count)
        with torch.no_grad():
            content = Image.open(os.path.join(test_root, str(content_path))).convert('RGB')
            content = transform(content).to(device).unsqueeze(0)

            # process input image
            state = content
            tic = time.time()
            encoder.init_hidden_state()
            for s in range(1, steps + 1):
                action, _, _, enc = encoder.sample(state)
                if echo is not None:
                    action = echo(action, s-1)
                rec = decoder(action, enc)
                state = rec
            toc = time.time()
            times.append(toc - tic)

            utils.save_image(os.path.join(save_dir, 'frame_{}.png'.format(str(count).zfill(4))), rec.cpu().data[0])
            # utils.save_image(os.path.join(save_dir, '{}_content.jpg'.format(str(count).zfill(5))), content.cpu().data[0])

        count += 1
        if num_store is not None:
            if count >= num_store:
                break

    print('finish')
    print('times:', np.mean(times))


if __name__ == "__main__":
    # ------------ device ---------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------------ load model ------------------------------
    ckpt_path = "./checkpoints/checkpoint_sketch.ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    encoder = GaussianPolicy(3, 64).to(device)
    decoder = Executor(64, 3).to(device)
    echo = Echo(64, 10).to(device)
    encoder.load_state_dict(checkpoint['policy_state_dict'])
    decoder.load_state_dict(checkpoint['executor_state_dict'])
    echo.load_state_dict(checkpoint['echo_state_dict'])
    # ------------ test dataset root -----------------------
    video_names = ['alley_2']
    for name in video_names:
        echo.init_hidden_state()
        print(name)
        test_root = "../datasets/MPI-Sintel/training/final/" + name
        # ------------ save path ------------------------------
        save_path = "./outputs/" + name
        test_only_product(test_root, device, encoder, decoder, echo,
                          steps=3, image_size=None, save_dir=save_path)

