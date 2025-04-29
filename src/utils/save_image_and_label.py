import cv2


def save(idx:str, aug_img, aug_labels, base_path):
    
    # Save augmented image
    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('data/images/'+base_path+idx+'.jpg', aug_img_bgr)

    # Save augmented label
    output_label_path = 'data/labels/'+ base_path +idx+'.txt'
    with open(output_label_path, 'w') as f:
        for label in aug_labels:
            f.write(' '.join(map(str, label)) + '\n')
