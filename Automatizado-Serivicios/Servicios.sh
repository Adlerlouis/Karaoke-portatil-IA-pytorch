#!/bin/bash
opcion=""
input_number=""
input_number2=""
input_text=""
input_text2=""
input_text3=""
input_text4=""
input_type=""
input_text5="#!/usr/bin/env python3"
node-red(){
read -p   "inicio del  servico node-red  1 cierra el servico 2" input_number
if (($input_number == 1 )); then
    node-red-start
    source Servicios.sh
fi
if (($input_number == 2)); then
    node-red-stop
    source Servicios.sh
fi    
}
Monitor_Serial(){
    sudo chmod 666 /dev/ttyACM0 
    sudo minicom -D /dev/ttyACM0 -b 115200
    source Servicios.sh
}

Manual(){
   cd  /home/adler/Manual_de_Ayuda
   cat Manual_ayuda.txt    
   cd ~
   source Servicios.sh
}


SSH(){
  sudo systemctl enable vncserver-x11-serviced.service
  sudo systemctl start vncserver-x11-serviced.service
  vncserver-virtual
  source Servicios.sh
}

salir(){
        exit
	exit 
}
IP(){
  ifconfig 
  source Servicios.sh
}

PYTHON(){
read -p " Pulse 1 para poder crear un script de python,  Pulse 2 para activar jupyter_notebook, Pulse 3 para activar entorno PYTHON " input_type

if (( $input_type == 1 )); then
    read -p  "Nombre del programa con extencion (.py) :" input_name
    cd   PYTHON3INTELIGENCEARTIFICAL
    touch $input_name
    chmod +x $input_name
    echo "creando archivo de cabezera : $input_name" input_text
    echo $input_text5 >> $input_name # redireccionamos el texto al archivo
    nvim $input_name
fi

if (( $input_type ==2 )); then 
cd PYTHON3INTELIGENCEARTIFICAL
 jupyter notebook
 cd ~
 source Servicios.sh
fi
if (( $input_type ==3 )); then
#creamos entorno virtual para ejecutar programas de python3 sin necesidad de tener que usar algun ide 
 python3 -m venv ~/.platformio-venv
 source ~/.platformio-venv/bin/activate


 cd ~
 source Servicios.sh 
fi
}
EXTRACT(){
   ./INTARTADLER.sh
   source Servicios.sh
}
Reset(){
read -p   "Desea Reiniciar el Server (y/N) ?: " input_text3
if [[ "$input_text3" == "y" ]]; then
    sudo reboot
fi
if [[ "$input_text3" == "N" ]]; then 
    source Servicios.sh
fi

}
Memoriaswap(){
sudo fallocate -l 12G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

source Servicios.sh
}
apagado(){
read -p   "Desea apagar el Server (y/N) ?: " input_text4
if [[ "$input_text4" == "y" ]]; then
    sudo poweroff
fi
if [[ "$input_text4" = "N" ]]; then 
    source Servicios.sh
fi
}
echo  "Bienvenido al servidor Adler"
echo "      Opcion A : Activar servicio de node-red
      Opcion B : Activar Monitor Serial
      Opcion C : Activar el servicio de VNC 
      Opcion D : Manual de Usuario
      Opcion E : Python3_INTELIGENCIA ARTFICIAL 
      Opcion F: Salir del Server
      Opcion G : Ver ip de Server
      Opcion H : Activar Servicio de  Borzoi IA
      Opcion I : Modificar memoria swap
      Opcion J : Reset Servidor
      Opcion K:  Apagar Servidor
                               "
      
echo "Si usted es Usuario invitado seleccione alguna opcion"
echo "Si usted es el administrador puede explorar los ficheros presionando ctl c"
echo "Si usted necesita reingresar al menu solo USE LA PALABRA "Servicios-cli" "
echo "Si quiere activar el Servicio de Borzio IA y se salio del menu use la palabra Borzoi-cli"
echo "Si necesita ayuda presione F para ver descripcion de los servicios "
echo "sudo systemctl disable udisks2
      sudo systemctl stop udisks2"
read -p  "Escoge una opcion de el munu [A-Z] :" opcion

case $opcion in 
     "A") node-red;;
     "B") Monitor_Serial;;
     "C") SSH;;
     "D") Manual;;
     "E") PYTHON;;
     "F") salir;;
     "G") IP;;
     "H") EXTRACT;;
     "I") Memoriaswap;;
     "J")Reset;;
     "K")apagado;;
       *)  ;;
esac


