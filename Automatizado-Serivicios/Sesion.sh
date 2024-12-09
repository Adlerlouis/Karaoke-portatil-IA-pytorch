#!/bin/bash
# Manipulacion de archivos 
input_type=""
input_text="#!/usr/bin/env python3"
input_name=""
#crea programas de python con el nombre y editar sin problema 
#creamos una opcion para poder activar el anaconda o solo un archivo de python

read -p "Pulse 1 para activar ANACONDA, Pulse 2 para poder hacer un script python, Pulse 3 para activar  Jupyter Notbook:" input_type
if (( $input_type == 1)); then
    anaconda-navigator
fi    
if (( $input_type == 2)); then
    read -p  "Nombre del programa con extencion (.py) :" input_name
    touch $input_name
    chmod +x $input_name
    echo "creando archivo de cabezera : $input_name" input_text
    echo $input_text >> $input_name # redireccionamos el texto al archivo
    nvim $input_name
fi
if (( $input_type == 3)); then
    jupyter notebook
fi 
