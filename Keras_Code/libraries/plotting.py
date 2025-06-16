import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def add_epoch_counter_to_df(df):
    sort_list = []
    i = 0
    while i < len(df):
        sort_list.append(i)
        i += 1
    df['epochs'] = sort_list
    return df


def class_all(df, epochs, samples, time, name):
    # If model begins with Conv:
    '''sns.relplot(df, x='epochs', y='Accuracy', kind='line')  # 'scatter'
    sns.relplot(df, x='epochs', y='val_Accuracy', kind='line')
    sns.relplot(df, x='epochs', y='loss', kind='line')
    sns.relplot(df, x='epochs', y='val_loss', kind='line')'''
    plt.plot(df['epochs'], df['Accuracy'], 'y', label='Training ACC')
    plt.plot(df['epochs'], df['val_Accuracy'], 'r', label='Validation ACC')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Epochs: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, df['Accuracy'].mean(), tex)
    plt.legend()
    plt.show()


def class_all_sm(df, epochs, samples, time, name):
    # If model begins with Dense:
    '''sns.relplot(df, x='epochs', y='accuracy', kind='line')  # 'scatter'
    sns.relplot(df, x='epochs', y='val_accuracy', kind='line')
    sns.relplot(df, x='epochs', y='loss', kind='line')
    sns.relplot(df, x='epochs', y='val_loss', kind='line')'''
    plt.plot(df['epochs'], df['accuracy'], 'y', label='Training ACC')
    plt.plot(df['epochs'], df['val_accuracy'], 'r', label='Validation ACC')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Epochs: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, df['accuracy'].mean(), tex)
    plt.legend()
    plt.show()


def class_acc_only(df, epochs, samples, time, name):
    # sns.relplot(df, x='epochs', y='accuracy', kind='line')
    plt.plot(df['epochs'], df['accuracy'], 'y', label='Training ACC')
    plt.plot(df['epochs'], df['val_accuracy'], 'r', label='Validation ACC')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('ACC')

    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Epochs: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, df['accuracy'].mean(), tex)
    plt.legend()
    plt.show()


def regression_all(df, epochs, samples, time, name):
    '''sns.relplot(df, x='epochs', y='loss', kind='line')
    sns.relplot(df, x='epochs', y='mae', kind='line')
    sns.relplot(df, x='epochs', y='val_loss', kind='line')
    sns.relplot(df, x='epochs', y='val_mae', kind='line')'''
    plt.plot(df['epochs'], df['mae'], 'y', label='Training MAE')
    plt.plot(df['epochs'], df['val_mae'], 'r', label='Validation MAE')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('MAE')

    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Epochs: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, df['mae'].mean(), tex)
    plt.legend()
    plt.show()


def regr_mae_only(df, epochs, samples, time, name):
    # sns.relplot(df, x='epochs', y='mae', kind='line')
    plt.plot(df['epochs'], df['mae'], 'y', label='Training MAE')
    plt.plot(df['epochs'], df['val_mae'], 'r', label='Validation MAE')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Epochs: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, df['mae'].mean(), tex)
    plt.legend()
    plt.show()


def multiple_plots(df, epochs, samples, time, name):
    plt.plot(df['epochs'], df['mae'], 'y', label='Training MAE')
    plt.plot(df['epochs'], df['val_mae'], 'r', label='Validation MAE')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Epochs: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, df['mae'].mean(), tex)
    plt.legend()
    plt.show()


def regr_test_plot(df, epochs, samples, time, name):
    plt.plot(df['epochs'], df['mae'], 'y', label='Test MAE')
    plt.title(name)
    plt.xlabel('Networks')
    plt.ylabel('MAE')
    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Networks: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, df['mae'].mean(), tex)
    plt.legend()
    plt.show()


def class_mult_plots(df, epochs, samples, time, name):
    plt.plot(df['epochs'], df['Accuracy'], 'y', label='Training Accuracy')
    plt.plot(df['epochs'], df['val_Accuracy'], 'r', label='Validation Accuracy')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Epochs: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text(epochs, df['Accuracy'].mean(), tex)
    plt.legend()
    plt.show()


def class_networks(df, epochs, samples, time, name):
    plt.plot(df['epochs'], df['accuracy'], 'y', label='Test ACC')
    plt.title(name)
    plt.xlabel('Networks')
    plt.ylabel('ACC')

    text_3 = ('Time: ' + str(time) + 's')
    text_2 = ('Samples: ' + str(samples) + '\n')
    text = ('Networks: ' + str(epochs) + '\n')
    tex = text + text_2 + text_3
    plt.text((len(df)/5), df['accuracy'].mean(), tex)
    plt.legend()
    plt.show()


def preds_for_plots(prediction, label):
    # print(len(prediction))
    j = 0
    # print(prediction[0])
    # print(label[0])
    arg_pred = np.argmax(prediction, axis=1)
    arg_lab = np.argmax(label, axis=1)
    # print(arg_pred[0])
    # print(arg_lab[0])
    # print(prediction[1], label[1], arg_pred[1], arg_lab[1])
    for i in range(len(prediction)):
        if arg_pred[i] == arg_lab[i]:
            j += 1
    print(j)
    test_acc = j/len(prediction)
    return test_acc
