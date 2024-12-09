#!/bin/bash
input_text3=""
input_text4=""
opcion=""

extraccion(){
	
        cd /home/adler/PYTHON3INTELIGENCEARTIFICAL
	echo "INCIANDO"
	./VENTANA.py
	cd ~
	./INTARTADLER.sh
}
Preoproc(){
	cd /home/adler/PYTHON3INTELIGENCEARTIFICAL
        echo "Bienvenido al preprocesamiento"
	./Preproc.py
        ./Preproc2.py
        cd ~
        ./INTARTADLER.sh

	}
Espectogramas(){

   cd /home/adler/PYTHON3INTELIGENCEARTIFICAL
  taskset -c 0-3 ./Librosa_1.py	
   cd ~
  ./INTARTADLER.sh
}
Ver_espectograma(){
	cd /home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/img 
	taskset -c 0 gthumb 

}
CargarAudio(){
 ./interfazPY.sh
 ./INTARTADLER.sh
}
SeparadorAudio(){
input_OP=""
input_OP2="/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/AUDIOENT"
input_OP3=""
input_OP4=""
cd /home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/PROYECTO/Prototipo-Codigp-proy-2/demucs
echo "Bienvienido al servicio de Separacion de audios"
read -p "Ingrese la cancion a separar (.mp3)" input_OP3
taskset -c 0-3 python -m demucs.separate --two-stems=vocals "$input_OP2/$input_OP3" 
cd /home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/PROYECTO/Prototipo-Codigp-proy-2/demucs/separated/htdemucs
echo "The process has Finished !!!!"
read -p "Ingrese De nuevo el nombre de archivo para ver el resultado  :" input_OP4
vlc "$input_OP4"
cd ~
./INTARTADLER.sh 
}
Salir(){
   source Servicios.sh
}
Compresion_Archivos(){
input_opcion=""
input_directorio=""
input_number=""
input_directorio_2=""
input_mont=""
echo "Bienvenido al servicio de Compresion de Carpetas !!!"
read -p "Opcion [1] Comprimir carpeta:  Opcion [2] Descomprimir archivo : Opcion [3] Salir del programa" input_number

if (($input_number== 1)); then
cd ~/PYTHON3INTELIGENCEARTIFICAL/DISC
read -p "Inserte el nombre de la carpeta extencion (.zip) :" input_directorio
read -p "Inserte el nombre de la carpeta a comprimir :" input_opcion
if [ ! -e "$input_opcion" ]; then
echo " CARPETA NO VALIDA !! "
   exit 1 
fi
 taskset -c 0-3 zip -r "$input_directorio" "$input_opcion"
cd ~
./INTARTADLER.sh
fi

if (($input_number ==2 ));then 
cd ~/PYTHON3INTELIGENCEARTIFICAL
read -p "Inserte la carpeta a descomprimir :" input_directorio_2
taskset -c 0-3 unzip "$input_directorio_2"
cd ~		
./INTARTADLER.sh
fi 

if  (($input_number ==3 ));then  
cd ~ 
./INTARTADLER.sh
fi
}

MONTAJE(){
sudo systemctl stop udisks2
sudo systemctl disable udisks2
read -p "Opcion [1] Montar Disco:  Opcion [2] Desmontar disco:" input_mont
if (($input_mont== 1)); then
sudo mount -o uid=1000,gid=1000,umask=0000 /dev/sda3 /home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC
fi
if (($input_mont== 2)); then
sudo umount /home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC
fi
}

Conexion(){
bluetoothctl 
./INTARTADLER.sh
}
Grabacion(){
cd ~/PYTHON3INTELIGENCEARTIFICAL
./Entrada_Audio.py
cd ~/PYTHON3INTELIGENCEARTIFICAL/DISC/AUDIOENT
vlc
cd ~
./INTARTADLER.sh

}
borzoiPlus(){
./Automatizado-IA.sh
}
echo "
 ____                      _   ___    _     
| __ )  ___  _ __ _______ (_) |_ _|  / \    
|  _ \ / _ \| '__|_  / _ \| |  | |  / _ \   
| |_) | (_) | |   / / (_) | |  | | / ___ \  
|____/ \___/|_|  /___\___/|_| |___/_/   \_\ 
"
echo "   
                                                  ..:. .:  .:... .                        
                                                .~7?JJ?J?7?~^^^~~?7.                      
                                               .^~~^~!?!7?7~^^^^^~?~.                     
                                                   .^?J??7!!??~^^^~7~                     
                                                   .:JY55!~!PP?~^^^?J                     
                                                    :7YP7~~~!7J?~^^7!                     
                                        .           :^77!~~!!~!7~^^!!                     
                              .::^^^^~!!77!^~^^~:.. :!!7!~~!?7~~~~^~!                     
                           .~77!~~^^^^^^J7?7!~777~!!????!~!!7J?^^~~^!                     
                         .^?7!^^^^^^^~^77!~^~~!~~!7777?J77!~7JY?^^~~!:                    
                        .!!~^:^~~~~!!~~J!!~~~~~~~!?!7!7JJY?7~?JY7~~~~!                    
                       :77^^:^~!!~~!~^!Y!!~7~~~~~!77?7!!?YYY?????777!?:                   
                       !7^^^^~~~~~~!~~7J~!~!7~~~~~~77?7!77?JJYYJ7?7JJY7                   
                      ~7!^^^~~~~~!?Y7!!?~!^~!7!~~~~~~!?7!77???JYY??7??^                   
                     .7!~^^~~~~~?5PPP7!?77~~~!!~~~!~~!!?????7??J???~                      
                     :77~~~~~~~7YP55P5?7J!7^~~~!~~7~~!^~~!7?JY??7?Y^                      
                     :7?~~!~~~!?YYJJJ55JP??~~~~!!!!~~!~~~~~!7YY77?Y^                      
                     ^J7~~~~~!7!JYJJ?JY55JYJ!~~7!~~~~!!~~~!!!!?7~~!~                      
                     ~J7~~~~!?J^!JY?JJ?JJJ?!J!~??~~~~~!~~~~~~~~!!~!!                      
                    .!J7!~~~7JY~:?5JJJJ??Y? ~Y?757~~~^~~~~7~~~7??77:                      
                     :77~~~~?YY^ ^JYYYJJJ5^  .!5Y7~~~^~~!J?77Y5?77.                       
                     ^Y7!~~!?Y?: .!55YJJJY:    ?~~~7!~~7YJJYYJJ~7:                        
                     ^?YY7!~?YY?^  :75J??7^     7~~~!7JJJJJY7777~?.                        
                  .^~!?Y?!!55YJ:   :7J??7      ~!~~~!7. .:J?77~~?.                        
                 ^!!~~~JJ!?5JJJ.    ~?7?!      ^!~~7!~    ~Y7!~~J.                        
            .:^~!77!~!7JY!JJJY7     7!!?:      ^?^~?7:     7~~~~Y.                        
          :~777?!~!~!7?JY!YJ5?^   .!?~!~       ^?~~7?:     ^!~~~J.                        
        .~??!77~~~~^77JYJ!Y?J:    ^YY!!:       :?~~~Y^     .!~~~?:                        
       .^!J7~!~~!!~!JJYG7!Y~.      ^5?7.       :?~!~?^      !~~~7~                        
       ..7J7!~!!!7?JY5J577!.        ~?!~       ^?~7!~!      ^7~~!?                        
         ^???!???JYJY?^~7!^         :?!7.      !7^7!~~       !~!?J.                       
         ..~7?!?JY?7~.  !~!..::::::::777!....  77^~!~~       !!^7Y:                       
       ..:::^~~~!7!~~^~~J~??777777777J7!Y7!~~^:J!~^7!^       ~7^~!7.                      
      ..:^^~~!!!777??JJJ?!J5JJJJJJ???5Y?77?YJ7!5~~^!7:       ^!^~^7^                      
       .:^~~!!777???JJY5JJY5YYYJJJJJJJY55Y555J?57!~!?~::.....^77!!J~::....                
        ..:^~~~~!!!!77?JJYJJJJ??????JJJJYYYYYYJY?!!?Y??7777777?5JJYJ???!~^^:.             
          ..:::::::....::::^^~~!!!!!!77?JJJYYYYY5??Y5YYYYYYYYYY5PPY?JPGY?7~^:...          
                               .....::^~!7?JJJY55Y5YYPY5YYYYYYYYY55555YJ?7!~^:::..        
                                     .:^~!7?JJJJYYYY5YYYYYYYYJJJJJJJ??7!!~^^^:::..        
                                     ..:^^~!7??????JJJJJJ?????777!!~~~^^^:::::...         

                                   "
echo "      Opcion A : Visualizado del dataset
      Opcion B : Valdacion y preprocesamiento
      Opcion C : Generar espectogramas 
      Opcion D : Importar Audio
      Opcion E : Ver espectogramas 
      Opcion F : Compresion de carpetas
      Opcion G : Montaje de disco 
      Opcion H : IA-Separador-Audio
      Opcion I : Habilitar el bluetooth
      Opcion J : Grabar Audio-Telefono-TIEMPO-REAL
      Opcion K : IA-Separador-Audio-Tiempo-Real
      Opcion L : Salir del programa
      "


read -p "Escoge una opcion [A-Z]" opcion
case $opcion in
	"A")extraccion;;
	"B")Preoproc;;
	"C")Espectogramas;;
	"D")CargarAudio;;
	"E")Ver_espectograma;;
	"F")Compresion_Archivos;;
	"G")MONTAJE;;
	"H")SeparadorAudio;;
	"I")Conexion;;
	"J")Grabacion;;
	"K")borzoiPlus;;
	"L")Salir;;
          *) ;;
esac




