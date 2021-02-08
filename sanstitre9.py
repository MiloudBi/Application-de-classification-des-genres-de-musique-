

"""Premier exemple avec Tkinter.

On crée une fenêtre simple qui souhaite la bienvenue à l'utilisateur

"""

# On importe Tkinter
import sklearn as skl
from extract import compute_feature
from tkinter import *
from tkinter.filedialog import askopenfilename
from sanstitre6 import feature
from sklearn import preprocessing
from keras.models import load_model
model = load_model('model.h5')
import numpy as np
import pandas as pd

# On crée une fenêtre, racine de notre interface
fenetre = Tk()
fenetre.geometry("500x200")
x=feature()
#print(tracks)


def predict(xe,x):
    """
    
    #xe=np.concatenate((xe, feature()),axis=0)
    print(xe)
    xe = preprocessing.normalize(tracks[:,4:])
    acp=skl.decomposition.PCA(250)
    xe = acp.fit_transform(xe)
    
    #xe = acp.fit_transform(xe)
    print(xe[0].shape)
    """
    p=80
    #x=feature()
    #print(x)
    xm=compute_feature(xe).reshape((1,518))
    #print(xm)
    x=np.concatenate((xm, x),axis=0)
    #print(x)
    
    acp=skl.decomposition.PCA(p)
    x = acp.fit_transform(x)
    x = preprocessing.normalize(x)
    list_genre=['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']
    list_index=[3 ,6, 2, 1, 7, 5, 0, 4]
    return  list_genre[list_index.index(np.argmax(model.predict(x)[0]))]

    
def chooseFile():
    champ_label2["text"]='veuillez patienter '
    filename = askopenfilename()
    champ_label2["text"]=predict(filename,x)
    
    
# On crée un label (ligne de texte) souhaitant la bienvenue
# Note : le premier paramètre passé au constructeur de Label est notre
# interface racine


cadre = Frame(fenetre, width=400, height=180, borderwidth=1)
cadre.pack(side=TOP)

btn = Button(cadre, text="choisir music",command=chooseFile)
btn.pack(side="left")
cadre.grid(padx=20, pady=50)

champ_label2 = Label(cadre, text="resultat")
champ_label2.pack(side="left")



# On démarre la boucle Tkinter qui s'interompt quand on ferme la fenêtre
fenetre.mainloop()
print("termine")