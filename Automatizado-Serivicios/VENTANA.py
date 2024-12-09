#!/usr/bin/env python3
import os 
from tkinter  import *
from PIL import ImageTk,Image
from tkinter import ttk
import sqlite3

def function():
    ventana2 = Tk();
    ventana2.title("Base de datos ")
    #label= Label(ventana2,text="Comenzando extraccion ")
    ventana2.geometry("600x400") #crea la ventana de la base de datos
    conn=sqlite3.connect('BORZOI_IA_DATABASE.db')
    cursor=conn.cursor()
     # Crear la tabla si no existe
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS BORZOI_IA_TABLA (
            name TEXT UNIQUE,
            followers INTEGER
        )
    ''')
    #instrucciones para crea la ventana de el contenido de datos 
    #creamos una base de datos NUMERO DE PRUEBA 1 Y DATOS NECESARIOS  
    tree = ttk.Treeview(ventana2, columns=("name", "followers"), show="headings", height=10)
    #configuro las columnas y filas de la base de datos 
    tree.heading("name",text="Nombre de muestra")
    tree.heading("followers",text="Tipo de multimedia")
    #tree.heading("subs",text="Datos recibidos")
    #creamos las columnas de la tabla y desplegamos la tabla para el sistema 
    tree.column("name",width=200)
    tree.column("followers",width=400)
    
    #instrucciones para poder subir datos a la base de datos 

    conn.commit() #instrucion para guardar cambios en la base de datos
    tree.pack(pady=20)
    samples ='./samples/Audios'
    #crea elemetos de la base de datos
    data_files = [file for file in os.listdir(samples) if '.m4a' in file]
     # Si no existe, insertar el nuevo registro
    #if result[0] == 0:
     #   cursor.execute("INSERT INTO BORZOI_IA_TABLA (name, followers) VALUES (?, ?)", (file, f"{samples}/{file}"))

        #cursor.execute("INSERT INTO BORZOI_IA_TABLA (name,followers) VALUES (?,?)",(file,f"{samples}/{file}"))
    #CREAR UN CICLO FOR DE N MULTIMEDIAS QUE SE VAN A PODER INSERTAR 
    #cursor.execute("INSERT INTO BORZOI_IA_TABLA VALUES ('Prueba 2','muestra.mp3')") 
    #cursor.execute("DELETE FROM BORZOI_IA_TABLA WHERE name ='Prueba 2'") #elimina elementos de la base de datos
#    cursor.execute("DELETE INTO BORZOI_IA_TABLA WHERE (name,followers) VALUES (?,?)",(file,f"{samples}/{file}"))
    #cursor.execute("DELETE FROM BORZOI_IA_TABLA WHERE name = M1.m4a AND followers = ?", (file, f"{samples}/{file}"))amples) if '.m4a' in file]
    for file in data_files:
         # cursor.execute("SELECT COUNT(*) FROM BORZOI_IA_TABLA WHERE name = ?", (file,))
        cursor.execute("INSERT OR IGNORE INTO BORZOI_IA_TABLA (name, followers) VALUES (?, ?)", (file, f"{samples}/{file}"))
    #cursor.execute("DELETE FROM BORZOI_IA_TABLA")
    conn.commit()
    


