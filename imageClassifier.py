import pandas
import imageio
import sys
import numpy
import glob
import imutils
import os

numpy.set_printoptions(threshold=sys.maxsize)
# numpy print max width

final_df = pandas.DataFrame()

folder_path_building = "question2/part_2/training_data/image_dataset/buildings"
folder_path_dogs = "question2/part_2/training_data/image_dataset/dogs"
folder_path_faces = "question2/part_2/training_data/image_dataset/faces"

result = []
result_extended = []

file_paths = []
file_paths1 = glob.glob(os.path.join(folder_path_faces, "*"))
file_paths2 = glob.glob(os.path.join(folder_path_building, "*"))
file_paths3 = glob.glob(os.path.join(folder_path_dogs, "*"))
file_paths.append(file_paths1)
file_paths.append(file_paths2)
file_paths.append(file_paths3)

for i in range(0, 3):

    for file_path in file_paths[i]:
        print(file_path)
        imimage = imageio.v2.imread(file_path)
        imimage_rotate_positive_5 = imutils.rotate(imimage, angle=5)
        imimage_rotate_negative_5 = imutils.rotate(imimage, angle=-5)
        imimage_enlarge = imutils.resize(imimage, width=30)
        imimage_enlarge = imimage_enlarge[1:29, 1:29]
        imimage = imimage.flatten()
        imimage_rotate_positive_5 = imimage_rotate_positive_5.flatten()
        imimage_rotate_negative_5 = imimage_rotate_negative_5.flatten()
        imimage_enlarge = imimage_enlarge.flatten()
        result.append(imimage)
        result_extended.append(imimage_rotate_positive_5)
        result_extended.append(imimage_rotate_negative_5)
        result_extended.append(imimage_enlarge)
        
    result_df = pandas.DataFrame(result)
    result_df = result_df.add_prefix('pixel_')
    result_df['digit']=i
    result_df['extended']=0
    result_extended_df = pandas.DataFrame(result_extended)
    result_extended_df = result_extended_df.add_prefix('pixel_')
    result_extended_df['digit']=i
    result_extended_df['extended']=1

    final_df = pandas.concat([result_df, result_extended_df, final_df], axis=0)


print(final_df.shape)
print(final_df)

# save to csv file
final_df.to_csv("dataset_extended.csv", index=False)