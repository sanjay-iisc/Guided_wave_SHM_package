import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('.\Plotting_style\science.mplstyle')
def figureplot(x,y,ax=None,**keyword):
        A,XLABEL,YLABEL=[],'F[Hz]','[A.U]'
        for key, value in keyword.items():
            if key=='xlabel':
                XLABEL=keyword[key]
                A.append(key)
            elif key=='ylabel':
                YLABEL=keyword[key]
                A.append(key)
            elif key=='title':
                Title=keyword[key]
                A.append(key)
            elif key=='filename':
                FileName=keyword[key]
                A.append(key)
            elif key=='ylim':
                Ylim=keyword[key]
                A.append(key)
            elif key=='path':
                Path=keyword[key]
                A.append(key)

            
        for a in A:
            keyword.pop(a) 
        ax.plot(x,y,**keyword )
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(YLABEL)
        ax.legend()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.grid(True)
        for a in A :
            if a == 'title':
                ax.set_title(Title, fontsize=10)
            elif a== 'ylim':
                ax.set_ylim(Ylim)
            elif a =='filename':
                try:
                    plt.savefig(Path+FileName+'.png')
                except:
                    raise Exception('path for saving has not been given')