#!/bin/bash

borzoiPlus(){
#este es la automatizacion de toda la ia desde la entrada hasta la salida 
input_OP2="$HOME/PYTHON3INTELIGENCEARTIFICAL/DISC/AUDIOENT"
input_OP3=""
input_OP5="/$HOME/PYTHON3INTELIGENCEARTIFICAL/DISC/PROYECTO/Prototipo-Codigp-proy-2/demucs/separated/htdemucs"
cd ~/PYTHON3INTELIGENCEARTIFICAL
./Entrada_Audio.py
cd ~/PYTHON3INTELIGENCEARTIFICAL/DISC/AUDIOENT
echo "Confirmacion del audio "

 echo "Grabacion guardada"
# Capturar el archivo mas reciente en la carpeta
ultimo_archivo=$(ls -t "$input_OP2"/*.mp3 2>/dev/null | head -n 1)

# Verificar si encuentra  algun archivo
if [ -n "$ultimo_archivo" ]; then
    echo "Archivo más reciente: $ultimo_archivo"

#    echo "Reproduciendo el archivo con VLC..."
 #   vlc "$ultimo_archivo"
else
    echo "No se encontraron archivos .mp3 en $base_dir"
fi


cd /home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/PROYECTO/Prototipo-Codigp-proy-2/demucs
# Capturar el archivo más reciente generado por VLC
ultimo_archivo=$(ls -t "$input_OP2"/*.mp3 | head -n 1)
#input_OP2=$(ls -t "$input_OP3"/*.mp3 2>/dev/null | head -n 1)
echo "Bienvienido al servicio de Separacion de audios"
taskset -c 0-3 python -m demucs.separate --two-stems=vocals "$ultimo_archivo" 
cd /home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/PROYECTO/Prototipo-Codigp-proy-2/demucs/separated/htdemucs
echo "The process has Finished !!!!"
# Capturar el archivo más reciente generado por el separador y proyectamos en VLC
input_OP3=$(ls -t "$input_OP5" 2>/dev/null | head -n 1)
# Verificar si encuentra  algun archivo
    vlc "$input_OP3"
cd ~
./INTARTADLER.sh 

}

borzoiPlus
