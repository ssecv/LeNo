import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from model_ours import GateNet
torch.manual_seed(2018)
import time
from test_data import test_dataset
from saliency_metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm
from torch.nn import functional as F

img_transform = transforms.Compose([
    # transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()
Image.MAX_IMAGE_PIXELS = 1000000000

test_datasets = ['./data/HKU-IS']
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 1765 66.94

def main():
    net = GateNet().cuda()
    net.load_state_dict(torch.load('./weight/ours/26381.pth'))
    net.eval()

    with torch.no_grad():
        for root in test_datasets:
            sal_root = root + '/salmap/'
            gt_root = root + '/mask/'

            # fileNameList = root+ '/test.txt'
            # objNameList = []
            # for i in open(fileNameList, 'r'):
            #     objNameList.append(i.replace('\n', ''))

            root1 = os.path.join(root, 'image')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith('.jpg')]
            for idx, img_name in enumerate(img_list):
                # if img_name not in objNameList:
                #     continue

                img1 = Image.open(os.path.join(root, 'pgd/' + img_name + '.jpg')).convert('RGB')  # fgsm pgd AS image
                img_var = Variable(img_transform(img1).unsqueeze(0)).cuda()

                prediction = net(img_var)
                prediction = F.sigmoid(prediction)

                prediction = to_pil(prediction.data.squeeze(0).cpu())
                # prediction = prediction.resize((w, h), Image.BILINEAR)
                # prediction = prediction.resize((w, h), Image.NEAREST)
                prediction = np.array(prediction)
                Image.fromarray(prediction).save(sal_root + img_name + '.png')

            ########################### Evaluation #############################
            test_loader = test_dataset(sal_root, gt_root, )
            mae, fm, sm, em, wfm = cal_mae(), cal_fm(test_loader.size), cal_sm(), cal_em(), cal_wfm()

            for i in range(test_loader.size):
                print('predicting for %d / %d' % (i + 1, test_loader.size))
                sal, gt = test_loader.load_data()
                if sal.size != gt.size:
                    x, y = gt.size
                    sal = sal.resize((x, y))
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                gt[gt > 0.5] = 1
                gt[gt != 1] = 0
                res = sal
                res = np.array(res)
                if res.max() == res.min():
                    res = res / 255
                else:
                    res = (res - res.min()) / (res.max() - res.min())
                mae.update(res, gt)
                sm.update(res, gt)
                fm.update(res, gt)
                em.update(res, gt)
                wfm.update(res, gt)

            MAE = mae.show()
            maxf, meanf, _, _ = fm.show()
            sm = sm.show()
            em = em.show()
            wfm = wfm.show()
            print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(
                root, MAE, maxf, meanf, wfm, sm, em))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
