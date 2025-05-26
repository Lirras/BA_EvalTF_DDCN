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


def class_all(df):
    # If model begins with Conv:
    sns.relplot(df, x='epochs', y='Accuracy', kind='line')  # 'scatter'
    sns.relplot(df, x='epochs', y='val_Accuracy', kind='line')
    sns.relplot(df, x='epochs', y='loss', kind='line')
    sns.relplot(df, x='epochs', y='val_loss', kind='line')
    plt.show()


def class_all_sm(df):
    # If model begins with Dense:
    sns.relplot(df, x='epochs', y='accuracy', kind='line')  # 'scatter'
    sns.relplot(df, x='epochs', y='val_accuracy', kind='line')
    sns.relplot(df, x='epochs', y='loss', kind='line')
    sns.relplot(df, x='epochs', y='val_loss', kind='line')
    plt.show()


def regression_all(df):
    sns.relplot(df, x='epochs', y='loss', kind='line')
    sns.relplot(df, x='epochs', y='mae', kind='line')
    sns.relplot(df, x='epochs', y='val_loss', kind='line')
    sns.relplot(df, x='epochs', y='val_mae', kind='line')
    plt.show()


def multiple_plots(df):
    plt.plot(df['epochs'], df['mae'], 'y', label='Training MAE')
    plt.plot(df['epochs'], df['val_mae'], 'r', label='Validation MAE')
    plt.title('MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()
