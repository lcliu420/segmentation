import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class SegmentationMetricTracker(object):
    def __init__(self, num_classes, include_background=False, smooth=1e-5):
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth
        self.reset()

    def reset(self):
        class_count = self.num_classes if self.include_background else self.num_classes - 1
        self.intersections = np.zeros(class_count, dtype=np.float64)
        self.pred_sums = np.zeros(class_count, dtype=np.float64)
        self.target_sums = np.zeros(class_count, dtype=np.float64)
        self.unions = np.zeros(class_count, dtype=np.float64)

    def update(self, logits, target):
        prediction = torch.argmax(logits, dim=1)
        start_class = 0 if self.include_background else 1

        for metric_index, class_index in enumerate(range(start_class, self.num_classes)):
            pred_mask = prediction == class_index
            target_mask = target == class_index
            intersection = torch.logical_and(pred_mask, target_mask).sum().item()
            pred_sum = pred_mask.sum().item()
            target_sum = target_mask.sum().item()
            union = torch.logical_or(pred_mask, target_mask).sum().item()

            self.intersections[metric_index] += intersection
            self.pred_sums[metric_index] += pred_sum
            self.target_sums[metric_index] += target_sum
            self.unions[metric_index] += union

    def compute(self):
        dice_scores = []
        iou_scores = []

        for intersection, pred_sum, target_sum, union in zip(
            self.intersections, self.pred_sums, self.target_sums, self.unions
        ):
            if pred_sum == 0 and target_sum == 0:
                dice_scores.append(1.0)
                iou_scores.append(1.0)
                continue

            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            iou = (intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
            iou_scores.append(iou)

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        return mean_dice, mean_iou, dice_scores, iou_scores


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy().squeeze(0), label.squeeze(0).cpu().detach().numpy().squeeze(0)
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
