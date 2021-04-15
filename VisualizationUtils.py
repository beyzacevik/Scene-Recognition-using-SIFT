import cv2
def get_successful_images(y_test, y_predicted,x_test_df, images):
    pos_bedrooms = list()
    pos_mountains = list()
    pos_office = list()
    pos_highway = list()
    pos_kitchen = list()
    pos_livingroom = list()

    for i in range(len(y_test)):

        if y_test[i] == y_predicted[i]:
            if y_predicted[i] == 'Bedroom' and len(pos_bedrooms) < 5:
                img_idx = x_test_df.index.values[i]
                pos_bedrooms.append(images[img_idx])
            elif y_predicted[i] == 'Mountain' and len(pos_mountains) < 5:
                img_idx = x_test_df.index.values[i]
                pos_mountains.append(images[img_idx])
            elif y_predicted[i] == 'Office' and len(pos_office) < 5:
                img_idx = x_test_df.index.values[i]
                pos_office.append(images[img_idx])
            elif y_predicted[i] == 'Highway' and len(pos_highway) < 5:
                img_idx = x_test_df.index.values[i]
                pos_highway.append(images[img_idx])
            elif y_predicted[i] == 'Kitchen' and len(pos_kitchen) < 5:
                img_idx = x_test_df.index.values[i]
                pos_kitchen.append(images[img_idx])
            elif y_predicted[i] == 'LivingRoom' and len(pos_livingroom) < 5:
                img_idx = x_test_df.index.values[i]
                pos_livingroom.append(images[img_idx])

    return (pos_bedrooms, pos_office, pos_highway, pos_kitchen, pos_livingroom, pos_mountains)


def get_unsuccessful_images(y_test, y_predicted,x_test_df, images):
    neg_bedrooms = list()
    neg_mountains = list()
    neg_office = list()
    neg_highway = list()
    neg_kitchen = list()
    neg_livingroom = list()

    for i in range(len(y_test)):

        if y_test[i] != y_predicted[i]:
            if y_predicted[i] == 'Bedroom' and len(neg_bedrooms) < 5:
                img_idx = x_test_df.index.values[i]
                neg_bedrooms.append(images[img_idx])
            elif y_predicted[i] == 'Mountain' and len(neg_mountains) < 5:
                img_idx = x_test_df.index.values[i]
                neg_mountains.append(images[img_idx])
            elif y_predicted[i] == 'Office' and len(neg_office) < 5:
                img_idx = x_test_df.index.values[i]
                neg_office.append(images[img_idx])
            elif y_predicted[i] == 'Highway' and len(neg_highway) < 5:
                img_idx = x_test_df.index.values[i]
                neg_highway.append(images[img_idx])

            elif y_predicted[i] == 'Kitchen' and len(neg_kitchen) < 5:
                img_idx = x_test_df.index.values[i]
                neg_kitchen.append(images[img_idx])
            elif y_predicted[i] == 'LivingRoom' and len(neg_livingroom) < 5:
                img_idx = x_test_df.index.values[i]
                neg_livingroom.append(images[img_idx])

    return (neg_bedrooms, neg_office,neg_highway, neg_kitchen, neg_livingroom, neg_mountains )



def show_images(path_list, title, axs, no):

        im1 = cv2.imread(path_list[0])
        im2 = cv2.imread(path_list[1])
        im3 = cv2.imread(path_list[2])
        im4 = cv2.imread(path_list[3])
        im5 = cv2.imread(path_list[4])

        axs[0, no].imshow(im1)
        axs[0, no].axis("off")
        axs[1, no].imshow(im2)
        axs[1, no].axis("off")
        axs[2, no].imshow(im3)
        axs[2, no].axis("off")
        axs[3, no].imshow(im4)
        axs[3, no].axis("off")
        axs[4, no].imshow(im5)
        axs[4, no].axis("off")
        axs[0, no].set_title(title)

        return