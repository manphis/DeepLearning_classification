import numpy as np
import matplotlib.pyplot as plt
from numpy import array

def save_train_result(acc_list):
    print('save_train_result')
    x = array(np.arange(len(acc_list)))
    y = array(acc_list)
    plt.xlabel('training iterations (1/100)')
    plt.ylabel('Accuracy')
    plt.bar(x, y)
    plt.savefig('accuracy.png')
    plt.close()

    plt.xlabel('Accuracy')
    plt.ylabel('times')
    plt.hist(acc_list)
    plt.savefig('histogram.png')

    return

def plot_error_result(error_list, correct_list, predict_dataset, index_list):
    num_col = 4
    error_count = len(error_list)
    num_row = int(error_count/num_col) + 1

    if num_row == 1:
        num_row += 1

    print('error_list length = ', len(error_list), ' index_list length = ', len(index_list))

    fig, axs = plt.subplots(num_row, num_col)
    show_count = 0

    done = False

    for i in range(num_row):
        for j in range(num_col):
            index = index_list[show_count]
            axs[i][j].imshow(predict_dataset[index])
            title = error_list[show_count] + '-->' + correct_list[show_count]
            
            axs[i][j].set_title(title)
            axs[i][j].set_xticks(())
            axs[i][j].set_yticks(())

            show_count += 1
            if show_count == error_count:
                done = True
                break
        if done:
            break
    plt.show()
    return

# def plot_error_result(part_list, predict_dataset, result):
#     error_count = 0
#     error_list = []
#     for k in range(len(result)):
#         if result[k] != int(k/2):
#             error_count += 1
#             error_list.append(k)

#     num_col = 4
#     num_row = int(error_count/4) + 1
#     fig, axs = plt.subplots(num_row, num_col)
#     show_count = 0

#     for i in range(num_row):
#         for j in range(num_col):
#             index = error_list[show_count]
#             axs[i][j].imshow(predict_dataset[index])
#             title = part_list[result[index]] + '-->' + part_list[int(index/2)]
            
#             axs[i][j].set_title(title)
#             axs[i][j].set_xticks(())
#             axs[i][j].set_yticks(())

#             show_count += 1
#             if show_count == error_count:
#                 break
#     plt.show()

#     return

def plot_predict_result(part_list, predict_dataset, result):
    num_row = int(len(predict_dataset)/6)
    num_col = 6
    fig, axs = plt.subplots(num_row, num_col)

    for i in range(num_row):
        for j in range(num_col):
            index = i*num_col + j
            axs[i][j].imshow(predict_dataset[index])
            title = '' + part_list[result[index]]
            if result[index] != int(index/2):
                title += "(X)"
            
            axs[i][j].set_title(title)
            axs[i][j].set_xticks(())
            axs[i][j].set_yticks(())
    plt.show()

    return