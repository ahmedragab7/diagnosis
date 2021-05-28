
import tkinter
from PIL import Image
import os
import PIL
import glob
from tkinter import *
import time
import random, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.vision.all import *
P_g=0



def demo():
    tkinter.Label(window1,image=im2).place(x=-40,y=-20)
    lable=tkinter.Label(window1,image=impr).place(x=0,y=0)
    tkinter.Label(window1, text = "Loading... ",fg ="red" ,bg ='white smoke',font =("Helvetica", 15)).place(x = 450,y =100)

    window1.after(1100,result_fun)


def predict():
    global P_g
    global path_n
    path_t=os.path.join('/home/ahmed_ragab/Pictures/x_ray/',path_n)
    path=Path('/home/ahmed_ragab/Downloads/archive/chest_xray')
    #print(path.ls())
    data = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

    dls = data.dataloaders(path)
    dls.valid.show_batch(max_n=12, nrows=2)
    learn = cnn_learner(dls, resnet18, metrics=[error_rate,accuracy])
    modell=learn.load('model_1.h5')
    P=modell.predict(path_t)
    P_g=P[0]
    print(P_g)

user=0
passw=0
pname=''
national_g=''
age_g=0
gender_g='not_s'
selected_d=[]
list=0
path_n='1.jpeg'

#image = PIL.Image.open('/home/ahmed_ragab/Desktop/pr.jpg')
#resized_image = image.resize((100,100))
#resized_image.save('/home/ahmed_ragab/Desktop/prrs.png')


def selected_item():
    global selected_d
    global list
    # Traverse the tuple returned by
    # curselection method and print
    # correspoding value(s) in the listbox
    for i in list.curselection():
        print(list.get(i))
        selected_d.append(list.get(i))



def information_p():
    global list
    global pname
    global national_g
    global age_g
    global gender_g
    global selected_d
    selected_d=[]
    lable=tkinter.Label(window1,image=im2)
    lable.place(x=-40,y=-20)
    lable=tkinter.Label(window1,image=impr)
    lable.place(x=0,y=0)
    tkinter.Label(window1, text = "Wellcome Doctor Ahmed",font=("Tempus Sans ITC", 20),bg ='sky blue',fg ="red4").place(x= 1,y=90)
    tkinter.Label(window1, text = "Please Enter patient's Data :",fg ="blue" ,bg ='white smoke',font =("Helvetica", 15)).place(x = 420,y =100)
    tkinter.Label(window1, text = "Name:",fg ="blue" ,bg ='white smoke',font =("Helvetica", 15)).place(x = 420,y =170)
    name=tkinter.Entry(window1)
    name.place(x = 590, y = 170) # first input-field is placed on position 01 (row - 0 and column - 1)
    name.bind("<Return>",namefun)
    tkinter.Label(window1, text = "national number:",fg ="blue" ,bg ='white smoke',font =("Helvetica", 15)).place(x = 420,y =210)
    national_n=tkinter.Entry(window1)
    national_n.place(x = 590, y = 210) # first input-field is placed on position 01 (row - 0 and column - 1)
    national_n.bind("<Return>",national_nfun)
    tkinter.Label(window1, text = "age:",fg ="blue" ,bg ='white smoke',font =("Helvetica", 15)).place(x = 420,y =250)
    age=tkinter.Entry(window1)
    age.place(x = 590, y = 250) # first input-field is placed on position 01 (row - 0 and column - 1)
    age.bind("<Return>",age_fun)
    tkinter.Label(window1, text = "gender:",fg ="blue" ,bg ='white smoke',font =("Helvetica", 15)).place(x = 420,y =290)

    gender1 = tkinter.Checkbutton(window1, text='male',variable=var1, onvalue=1, offvalue=0, command=gender_fun,font =("Helvetica", 15),bg ='white smoke',fg ="blue").place(x=590,y=290)
    gender2 = tkinter.Checkbutton(window1, text='female',variable=var2, onvalue=1, offvalue=0, command=gender_fun,font =("Helvetica", 15),bg ='white smoke',fg ="deep pink").place(x=590,y=330)
    tkinter.Label(window1, text = "chronic diseases:",fg ="blue" ,bg ='white smoke',font =("Helvetica", 15)).place(x = 420,y =370)
    list = tkinter.Listbox(window1, selectmode = "multiple",width=20, height=7,selectbackground='red')
    list.place(x=590,y=370)
    # Inserting the listbox items
    x = ["Hypertension", "Diabetes", "Renal Failure", "Liver failure", "Liver failure","Eczema", "Heart failure"]
    for each_item in range(len(x)):
        list.insert(END, x[each_item])
        # coloring alternative lines of listbox
        list.itemconfig(each_item,bg = "sky blue" if each_item % 2 == 0 else "sky blue")
    button_widget = tkinter.Button(window1,text="revise the data and load the X_ray  ", command = x_ray_fun,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=500,y=600)


def load_x_ray():
    global imx
    lable=tkinter.Label(window1,image=imx)
    lable.place(x=490,y=300)

def final():
    global pname
    global national_g
    global age_g
    global gender_g
    global selected_d
    global P_g

    lable=tkinter.Label(window1,image=im2)
    lable.place(x=-40,y=-20)
    lable=tkinter.Label(window1,image=impr)
    lable.place(x=0,y=0)
    tkinter.Label(window1, text = "Wellcome Doctor Ahmed",font=("Tempus Sans ITC", 20),bg ='sky blue',fg ="red4").place(x= 1,y=90)
    tkinter.Label(window1, text = "Patient's Name:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="black").place(x = 420,y =100)
    tkinter.Label(window1, text =pname ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =100)
    tkinter.Label(window1, text = "national number:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="black").place(x = 420,y =120)
    tkinter.Label(window1, text =national_g ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =120)
    tkinter.Label(window1, text = "Patient's age:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="black").place(x = 420,y =140)
    tkinter.Label(window1, text =age_g ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =140)
    tkinter.Label(window1, text = "Patient's gender:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="black").place(x = 420,y =160)
    tkinter.Label(window1, text =gender_g ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =160)
    tkinter.Label(window1, text = "chronic diseases:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="black").place(x = 420,y =180)
    tkinter.Label(window1, text =selected_d ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =180)
    lable=tkinter.Label(window1,image=imx)
    lable.place(x=490,y=210)
    if P_g=='NORMAL':
        tkinter.Label(window1, text = pname+' s X_ray is',fg ="blue" ,bg ='white smoke',font =("Helvetica", 11)).place(x = 420,y =493)
        tkinter.Label(window1, text = 'Normal',fg ="green" ,bg ='white smoke',font =("Helvetica", 20)).place(x = 585,y =490)
        button_widget = tkinter.Button(window1,text="Print", command = final,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=490,y=630)
        button_widget = tkinter.Button(window1,text="Home", command = information_p,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=650,y=630)

    else:
        tkinter.Label(window1, text = pname+' has',fg ="blue" ,bg ='white smoke',font =("Helvetica", 11)).place(x = 420,y =493)
        tkinter.Label(window1, text = 'Acute pneumonia',fg ="red4" ,bg ='white smoke',font =("Helvetica", 19)).place(x = 585,y =490)
        #t2=string(t21+'so please revise the specialist')

        #t=string(t1+'has'+t2)
        tkinter.Label(window1, text ='and because '+pname+' has' ,fg ="blue" ,bg ='white smoke',font =("Helvetica", 11)).place(x = 420,y =525)
        tkinter.Label(window1, text =selected_d ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 599,y =525)
        tkinter.Label(window1, text ='so please revise the specialist' ,fg ="blue" ,bg ='white smoke',font =("Helvetica", 11)).place(x = 420,y =550)
        button_widget = tkinter.Button(window1,text="Print", command = final,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=490,y=630)
        button_widget = tkinter.Button(window1,text="Home", command = information_p,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=650,y=630)



def result_fun():
    global P_g
    global pname
    global selected_d
    lable=tkinter.Label(window1,image=im2)
    lable.place(x=-40,y=-20)
    lable=tkinter.Label(window1,image=impr)
    lable.place(x=0,y=0)
    tkinter.Label(window1, text = "The Report :",fg ="red" ,bg ='white smoke',font =("Helvetica", 19)).place(x = 450,y =100)
    predict()
    if P_g=='NORMAL':
        tkinter.Label(window1, text = pname+' s X_ray is',fg ="blue" ,bg ='white smoke',font =("Helvetica", 11)).place(x = 450,y =160)
        tkinter.Label(window1, text = 'Normal',fg ="green" ,bg ='white smoke',font =("Helvetica", 20)).place(x = 450,y =200)
        button_widget = tkinter.Button(window1,text="Print a Report", command = final,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=590,y=590)
        lable=tkinter.Label(window1,image=imx)
        lable.place(x=490,y=300)
    else:
        tkinter.Label(window1, text = pname+' has',fg ="blue" ,bg ='white smoke',font =("Helvetica", 11)).place(x = 450,y =160)
        tkinter.Label(window1, text = 'Acute pneumonia',fg ="red4" ,bg ='white smoke',font =("Helvetica", 19)).place(x = 450,y =200)
        #t2=string(t21+'so please revise the specialist')

        #t=string(t1+'has'+t2)
        tkinter.Label(window1, text ='and because '+pname+' has' ,fg ="blue" ,bg ='white smoke',font =("Helvetica", 11)).place(x = 450,y =240)
        tkinter.Label(window1, text =selected_d ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 610,y =240)
        tkinter.Label(window1, text ='so please revise the specialist' ,fg ="blue" ,bg ='white smoke',font =("Helvetica", 11)).place(x = 450,y =260)
        button_widget = tkinter.Button(window1,text="Print a Report", command = final,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=590,y=590)
        lable=tkinter.Label(window1,image=imx)
        lable.place(x=490,y=300)


def x_ray_fun():
    global pname
    global national_g
    global age_g
    global gender_g
    global selected_d
    global imx
    global path_n
    selected_item()
    lable=tkinter.Label(window1,image=im2)
    lable.place(x=-40,y=-20)
    lable=tkinter.Label(window1,image=impr)
    lable.place(x=0,y=0)
    tkinter.Label(window1, text = "Wellcome Doctor Ahmed",font=("Tempus Sans ITC", 20),bg ='sky blue',fg ="red4").place(x= 1,y=90)
    tkinter.Label(window1, text = "Patient's Name:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="blue2").place(x = 420,y =100)
    tkinter.Label(window1, text =pname ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =100)
    tkinter.Label(window1, text = "national number:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="blue2").place(x = 420,y =120)
    tkinter.Label(window1, text =national_g ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =120)
    tkinter.Label(window1, text = "Patient's age:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="blue2").place(x = 420,y =140)
    tkinter.Label(window1, text =age_g ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =140)
    tkinter.Label(window1, text = "Patient's gender:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="blue2").place(x = 420,y =160)
    tkinter.Label(window1, text =gender_g ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =160)
    tkinter.Label(window1, text = "chronic diseases:",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="blue2").place(x = 420,y =180)
    tkinter.Label(window1, text =selected_d ,font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="red4").place(x = 590,y =180)
    tkinter.Label(window1, text ="enter the X_ray name",font=("Tempus Sans ITC", 11),bg ='white smoke',fg ="blue2").place(x = 420,y =200)
    xr=tkinter.Entry(window1)
    xr.place(x = 590,y =200)
    xr.bind("<Return>",path)


    button_widget = tkinter.Button(window1,text="Load", command = load_x_ray,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=770,y=197)
    button_widget = tkinter.Button(window1,text="X_ray's Result", command = demo,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font =('BOLD',10),activeforeground ='white').place(x=590,y=590)









def fun(en1):
    global user
    if (en1.widget.get()=="ahmed"):
        user=1
    print(en1.widget.get())

def fun2(en2):
    global passw
    if (en2.widget.get()=="000"):
        passw=1
    print(en2.widget.get())

def login():
    global user,passw,var1
    if (user==1 and passw==1):
        information_p()

def namefun(name):
    global pname
    pname=name.widget.get()
    print(pname)

def path(xr):
    global path_n
    global imx
    path_n=xr.widget.get()
    total_path=os.path.join('/home/ahmed_ragab/Pictures/x_ray/',path_n)
    print(total_path)
    image = PIL.Image.open(total_path)
    resized_image = image.resize((270,270))
    resized_image.save('/home/ahmed_ragab/Pictures/x_ray/x1.png')
    imx=tkinter.PhotoImage(file = "/home/ahmed_ragab/Pictures/x_ray/x1.png")
    print(path_n)

def national_nfun(national_n):
    global national_g
    national_g=national_n.widget.get()
    print(national_g)

def age_fun(age):
    global age_g
    age_g=age.widget.get()
    print(age_g)

def gender_fun():
    global gender_g
    print(var1.get())
    if var1.get()==1:
        gender_g='male'
    if var2.get()==1:
        gender_g='female'

window1=tkinter.Tk()

var1 = IntVar()
var2 = IntVar()
window1.geometry("1000x700")
window1.title("pneumothorax Diagnosis")
im=tkinter.PhotoImage(file = "/home/ahmed_ragab/Pictures/hc.png")
im2=tkinter.PhotoImage(file = "/home/ahmed_ragab/Desktop/d3.png")
impr=tkinter.PhotoImage(file = "/home/ahmed_ragab/Desktop/prrs.png")
imx=0


lable=tkinter.Label(window1,image=im)
lable.place(x=-40,y=-20)
#canvas1 = tkinter.Canvas( window1, width = 1366,height = 768)
#canvas1.grid(row = 0, column = 0)

# Display image
#canvas1.create_image( 0, 0, image = im,anchor = "nw")
# pack is used to show the object in the window
#
tkinter.Label(window1, text = "Username", fg ="red4" ,bg ='sky blue',font ='BOLD').place(x = 10,y =10)#'username' is placed on position 00 (row - 0 and column - 0)
tkinter.Label(window1, text = "Password",fg ="red4" ,bg ='sky blue',font ='BOLD').place(x = 10,y =50) #'username' is placed on position 00 (row - 0 and column - 0)


# 'Entry' class is used to display the input-field for 'username' text label
en1=tkinter.Entry(window1)
en1.place(x = 110, y = 10) # first input-field is placed on position 01 (row - 0 and column - 1)
en1.bind("<Return>",fun)

en2=tkinter.Entry(window1)
en2.place(x = 110, y = 50) # first input-field is placed on position 01 (row - 0 and column - 1)
en2.bind("<Return>",fun2)

#tkinter.Label(window1, text = "Password").place(x= 2,y=2) #'password' is placed on position 10 (row - 1 and column - 0)

#tkinter.Entry(window1).grid(row = 3, column = 3) #second input-field is placed on position 11 (row - 1 and column - 1)

button_widget = tkinter.Button(window1,text="Login", command = login,activebackground = 'red4',fg = 'red4',bg ='white',highlightcolor ='white',font ='BOLD',activeforeground ='white')
button_widget.place(x = 10, y = 90)
window1.mainloop()

















