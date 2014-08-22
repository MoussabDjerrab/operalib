import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_err(obj,var,ensvar,X,y,sparcity=False,xscale='log'):
    """
    Given a model, a parameter and a set of variation.
    This function is plotted and errors tests drive (and the sparcity of C) depending on the value of the given parameter

        obj : our model
        var : our parameter
        ensvar : set of variation
        sparicty : boolean : do you want to plot the sparcity of C ?
        xscale : scale of x
    """
    # TODO : sparcity of group if partition not None
    ensvar.sort()
    _,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xscale(xscale)
    ax2.set_xscale(xscale)
    ax1.set_ylabel("error")
    ax1.set_xlabel(var)
    def f(x) :
        obj.setparam(var,x)
        obj.fit(X,y)
        tra = obj.score(X,y)
        tes = obj.crossvalidationscore(X,y)
        spar = float((obj.C == 0).sum())/obj.C.size * 100
        return (tra,tes,spar)
    train_err = []
    valid_err = []
    sparc  = []
    for x in ensvar :
        (tra,tes,spar) = f(x)
        train_err.append(tra)
        valid_err.append(tes)
        sparc.append(spar)
    ax1.plot(ensvar,train_err,'o-',color='g')
    ax1.plot(ensvar,valid_err,'s-',color='r')
    if sparcity :
        ax2.set_ylabel("sparcity of C (%)",color='#AFAFAF')
        ax2.plot(ensvar,sparc,'x-',color='#AFAFAF')
        for tl in ax2.get_yticklabels():
            tl.set_color('#AFAFAF')

    ax1.legend(['training error','testing error'],'best')

    xmin = ensvar[(np.array(valid_err)).argmin()]
    ymin = (np.array(valid_err)).min()
    (axm,_,aym,_)  = ax1.axis()
    ax1.plot([xmin,xmin],[aym,ymin],'--k')
    ax1.plot([axm,xmin],[ymin,ymin],'--k')


    return xmin



def matrix_to_graph(M,name=None,oriented=False,node_shape="o",node_size=1000,font_size=14,node_color="white"):
    """

    """
    G = nx.Graph(M)
    if name is None : nx.draw(G); return G
    labels = {}
    for i in range(len(name)) :
        labels[i] = str(name[i])
    nx.draw(G,labels=labels,with_labels=True,node_shape=node_shape,font_size=font_size,node_size=node_size,node_color=node_color)
    plt.show()

def plot_predicted_graph(G,H=None,name=None,node_shape="o",node_size=1000,font_size=14,node_color="white"):
    if H is None : return matrix_to_graph(G,name=None,node_shape="o",node_size=1000,font_size=14,node_color="white")
    G = nx.Graph(G)
    H = nx.Graph(H)
    I = nx.intersection(G, H)
    U = nx.compose(G, H)
    edge_color=[]
    edge_style=[]
    for edge in U.edges() :
        if edge in I.edges() :
            edge_color.append("black")
            edge_style.append("solid")
        elif edge in H.edges() :
            edge_color.append("black")
            edge_style.append("dotted")
        else :
            edge_color.append("red")
            edge_style.append("solid")
    labels = {}
    if name is not None :
        for i in range(len(name)) :
            labels[i] = str(name[i])
    else :
        labels = None
    nx.draw(U,labels=labels,with_labels=True,node_shape=node_shape,font_size=font_size,node_size=node_size,node_color=node_color,edge_color=edge_color,style=edge_style)
    plt.show()

def plot_time_series(TS):
    plt.plot(TS)
    plt.show()