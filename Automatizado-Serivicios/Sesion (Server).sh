#!/bin/bash
# sesion del server

opcion=0
read -p "Inicia sesion Opcion 1:Adler-Server Opcion 2:Rasberrypi2W :Opcion3: Adler-Server WIFI-CONEXION :" opcion
if (($opcion ==1 )); then 
    ssh -X adler@10.42.0.29
elif (($opcion ==2 )); then 
    ssh adler2wserver@192.168.1.78
elif (($opcion ==3 )); then 
    ssh -X adler@192.168.1.74

fi    
