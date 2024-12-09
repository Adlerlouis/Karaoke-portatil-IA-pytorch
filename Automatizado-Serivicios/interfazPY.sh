#!/bin/bash
input_number2=""
Copy(){
read  -p "Audio-cli  para importar  audio pulse 1  para ver audio pulse 2 para salir pulse 3  " input_number2
if (($input_number2 == 1 )); then 
   scp -r adler-llab@10.42.0.1:/home/adler-llab/Audios  ~/PYTHON3INTELIGENCEARTIFICAL/DISC/samples
   cd ~ 
   ./interfazPY.sh
fi
if (( $input_number2 == 2)); then 
   cd ~/PYTHON3INTELIGENCEARTIFICAL/DISC/samples
   ls -l
   cd ~ 
   ./interfazPY.sh 
fi

if (($input_number2 == 3)); then
   cd
   ./INTARTADLER.sh
fi
}
#corrermos la funcion  de importacion de audio 
Copy
