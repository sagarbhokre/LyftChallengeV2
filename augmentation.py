import cv2
import numpy as np


def rotate(image,
           angle, tx, ty, scale,
           interpolation=cv2.INTER_LINEAR,
           border_value=0):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    M[0, 2] = tx
    M[1, 2] = ty
    image = cv2.warpAffine(image, M, (w, h),
                           flags=interpolation, borderValue=border_value)
    return image


def rotate_both(image, label, label2, label3, p=0.5, ignore_label=1):
    if np.random.rand() > p:
        return image, label, label2, label3
    h, w = image.shape[:2]
    angle = np.random.uniform(-10, 10)
    tx = np.random.uniform(-w // 128, w // 128)
    ty = np.random.uniform(-h // 128, h // 128)
    scale = np.random.uniform(0.98, 1.02)
    image = rotate(image, angle, tx, ty, scale, border_value=(127, 127, 127))
    label = rotate(label, angle, tx, ty, scale,
                   interpolation=cv2.INTER_NEAREST, border_value=ignore_label)
    label2 = rotate(label2, angle, tx, ty, scale,
                   interpolation=cv2.INTER_NEAREST, border_value=ignore_label)
    label3 = rotate(label3, angle, tx, ty, scale,
                   interpolation=cv2.INTER_NEAREST, border_value=ignore_label)
    return image, label, label2, label3


def flip_both(image, label, label2, label3, p=0.5):
    if np.random.rand() > p:
        return image, label, label2, label3
    image = np.fliplr(image)
    label = np.fliplr(label)
    label2 = np.fliplr(label2)
    label3 = np.fliplr(label3)
    return image, label, label2, label3


def blur(image, k=7):
    image = cv2.GaussianBlur(image, (k, k), 0)
    return image


def blur_both(image, label, label2, label3, p=0.5, k=7):
    if np.random.rand() > p:
        return image, label, label2, label3
    image = blur(image, k=k)
    return image, label, label2, label3


def change_illumination(
        image, sat_limit=(-12, 12), val_limit=(-30, 30)):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    sat_shift = np.random.uniform(sat_limit[0], sat_limit[1])
    s = cv2.add(s, sat_shift)
    value_shift = np.random.uniform(val_limit[0], val_limit[1])
    v = cv2.add(v, value_shift)
    image = np.dstack((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def illumination_change_both(image, label, label2, label3, p=0.5):
    if np.random.rand() > p:
        return image, label, label2, label3
    image = change_illumination(image)
    return image, label, label2, label3


if __name__ == '__main__':
    np.random.seed(13)
    background_color = np.uint8([0, 0, 255])
    img = cv2.imread(
        'data/data_road/training/image_2/umm_000023.png')
    gt = cv2.imread(
        'data/data_road/training/gt_image_2/umm_road_000023.png', 1)
    gt = np.uint8(np.all(gt == background_color, axis=2))
    g2rgb = lambda img: np.dstack((255 * img, 0 * img, 255 * img))
    put_text = lambda img, text: cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 127, 127), 3, cv2.LINE_AA)  # noqa
    img_rot, gt_rot = rotate_both(img, gt, p=1.0, ignore_label=1)
    img_flip, gt_flip = flip_both(img, gt, p=1.0)
    img_blur, gt_blur = blur_both(img, gt, p=1.0, k=33)
    img_ill, gt_ill = illumination_change_both(img, gt, p=1.0)
    img_mixed, gt_mixed = rotate_both(img, gt, p=1.0, ignore_label=1)
    img_mixed, gt_mixed = flip_both(img_mixed, gt_mixed, p=1.0)
    img_mixed, gt_mixed = blur_both(img_mixed, gt_mixed, p=1.0, k=15)
    img_mixed, gt_mixed = illumination_change_both(img_mixed, gt_mixed, p=1.0)

    r0 = np.hstack((img, g2rgb(gt)))
    put_text(r0, "Orginal")
    r1 = np.hstack((img_rot, g2rgb(gt_rot)))
    put_text(r1, "Rotated")
    r2 = np.hstack((img_flip, g2rgb(gt_flip)))
    put_text(r2, "Flipped")
    r3 = np.hstack((img_blur, g2rgb(gt_blur)))
    put_text(r3, "Blurred (exaggerated)")
    r4 = np.hstack((img_ill, g2rgb(gt_ill)))
    put_text(r4, "Illumination changed")
    r5 = np.hstack((img_mixed, g2rgb(gt_mixed)))
    put_text(r5, "All previous")
    pastiche = np.vstack((r0, r1, r2, r3, r4, r5))
    fn = 'augmentation_methods.png'
    cv2.imwrite(fn, pastiche)
    print("Wrote img to `%s`" % fn)
