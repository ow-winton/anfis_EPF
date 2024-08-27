from matplotlib import pyplot as plt
def plot_sub(st, et, df_all, parm={}, length=25):
    df_plot = df_all.loc[st:et, ]
    num = len(parm)
    fig, ax = plt.subplots(num, 1, sharex="all", figsize=(length, num * 3))
    font_size = 15
    for i, p in enumerate(parm):
        if '实时电价' in p:
            fig = df_plot.loc[:,
                  [
                      '实时电价',
                      '日前电价',
                      '预测实时电价',
                      
                  ]].plot(title='实时电价预测',
                          ax=ax[i],
                          style=['-', '-', '--'],
                          linewidth=2.5,
                          fontsize=14,
                          grid=True,
                          secondary_y=[])
            fig.title.set_size(font_size)
            ax[i].fill_between(df_plot.index, df_plot['实时电价'], df_plot['预测实时电价'],
                               where=df_plot['实时电价'] > df_plot['预测实时电价'],
                               color="#f8f2e4")
        else:

            fig = df_plot.loc[:, parm[p]].plot(title=p,
                                               ax=ax[i],
                                               style=['-', '--', '--', '--', '--', '--', 'o', 'o'],
                                               linewidth=2.5,
                                               fontsize=14,
                                               grid=True,
                                               secondary_y=[])
            fig.title.set_size(font_size)