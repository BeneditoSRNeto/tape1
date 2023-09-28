from kivy.clock import mainthread
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
import numpy as np
import cv2
from camera4kivy import Preview
import json
import math
import csv 
from kivy.app import App
from kivy.storage.jsonstore import JsonStore

from android.permissions import Permission, request_permissions, check_permission
from android.storage import app_storage_path, primary_external_storage_path, secondary_external_storage_path

import os

from kivy.logger import Logger
from time import time
from time import process_time
 



class EdgeDetect(Preview):
    texto=[]
    aspect_metros_x = 0;
    #path=""
    fname=""
    store=None
    contador=0
    milliseconds = 0
    last_detect=0
    t1_start = 0
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzed_texture = None
        
        #self.path=App.get_running_app().user_data_dir+"/"
        
        perms = [Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE]
      
        if  self.check_permissions(perms)!= True:
            request_permissions(perms)    # get android permissions     
           
            exit()                        # app has to be restarted; permissions will work on 2nd start
        self.t1_start = process_time()
        #self.fname = os.path.join( primary_external_storage_path(),'testfile')
    ####################################
    # Analyze a Frame - NOT on UI Thread
    ####################################

    def analyze_pixels_callback(self, pixels, image_size, image_pos, scale, mirror):
        # pixels : analyze pixels (bytes)
        # image_size   : analyze pixels size (w,h)
        # image_pos    : location of Texture in Preview (due to letterbox)
        # scale  : scale from Analysis resolution to Preview resolution
        # mirror : true if Preview is mirrored
        
        rgba   = np.fromstring(pixels, np.uint8).reshape(image_size[1],
                                                         image_size[0], 4)
        # Note, analyze_resolution changes the result. Because with a smaller
        # resolution the gradients are higher and more edges are detected.
        
        # ref https://likegeeks.com/python-image-processing/
        #gray   = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        #blur   = cv2.GaussianBlur(gray, (3,3), 0)
        #edges  = cv2.Canny(blur,50,100)
        #rgba   = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGBA)
        # V=[]
        # filtro=self.FiltragemInicial(rgba)
        #rgba=self.distorcer_imagem(rgba)
        self.store = JsonStore(primary_external_storage_path()+'/Download/hello.json')
        self.carregaNovaPlanta()
        D=self.detect_lowScale(rgba,rgba)
        
        if D:
            for i in range(0,len(D),3): 
              P1=D[i]
              P2=D[i+1]
              ID=D[i+2]
              cv2.rectangle(rgba, (int(P2[0]),(P2[1])),(int(P1[0]),int(P1[1])) ,(255,0,0,255), 4)#azul
              self.escreve_id(rgba, str(ID),P1 )
           # self.escreve_id(rgba, primary_external_storage_path()+'/Download/hello.json',(100,100) )
            #self.store.put('tshirtman', name='Gabriel', age=27)
            #self.testwrite()
        else:
            
            txt=""
            binario=""
            sucesso=False
            linha2=[]
            Arr=self.getLine(rgba)
            (h, w) = rgba.shape[:2]
            rot = np.zeros((h, w, 4), dtype='uint8')
            linha=[]
            if not Arr is None:
                for lin in Arr: 
                    #print(lin['M'],lin['lines_list']) 
                    linha=np.dot(lin['M'],lin['lines_list']).astype(np.int32)
            
                    img_rot= cv2.warpAffine(rgba, lin['M'], (w, h))
            
                    rot=img_rot.copy()
                    encode=[]
                    if len(linha) > 0:
                        if rot.size>0:
                            x1=linha[0][0]-w
                            y1=linha[1][0]
                            x2=linha[0][1]+w
                            y2=linha[1][1]
                            cv2.line(rot,(linha[0][0],linha[1][0]),(x2,y2),(0,0,255,255),5)
                    
                            crop_img=img_rot[y1-100:y2+10,x1: x2]
                            h1, w1 = crop_img.shape[:2]
                   
                            if h1 > 0 and w1 > 0:
                    
                            #temp=FiltragemInicial(crop_img)
                                temp = cv2.cvtColor(crop_img, cv2.COLOR_RGBA2GRAY)
                            #gray   = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
                        
                                suave = cv2.GaussianBlur(temp, (7, 7), 0) # aplica blur
                                ret,thresh = cv2.threshold(suave,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            
                                thresh = cv2.bitwise_not(thresh)
                            
                                for l in range(0, h1-1):
                            
                                    lines= thresh[l]
                            
                                    for i in range(len(lines)):
                                        if lines[i] == 255:
                                            lines[i] = 1
                            
                                    string=self.array_to_string(lines)
                                    #print(l,"cadeia__ ",string)
                                    bar=self.bit_to_bars(string)
                                #print(l,"bar",bar)
                                    data_string=self.remove_quieteZone(bar)
                                
                                    ## varrer um por um e contar os padroes 
                                
                           
                                   # print(l,"limpa e valida",data_string)
                                    sbarras=self.remove_barras(data_string)
                                #print(l,"sem barras ",sbarras)
                                    squadrado=self.remove_quadrado(sbarras)
                                   # print(l,"sem quuadro ",squadrado)
                                    bits=self.encode_bars(squadrado)
                            
                                    if len(bits)>2:
                                        encode.append(bits)
                                
                                                  
                                    #print("akiiiiiiiiiiii",bits)
                        bit=self.valida_detect(encode)
                       # print(bit)
                        if(len(bit) > 2):
                            cont_tm,linha2=self.lerTMToDraw(bit)
                            if not linha2:
                                break               #print("Detectou em escal maior : ",bit)
                                sucesso=True
                            
                            else:
                                binario=str(bit)
                                sucesso=False
        
            #binario=self.detect_HighScale(rgba)
            #print(binario)
                                txt=binario
                                posicao=(100,100)
                                cor=(0,0,220,255)
                                fonte = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(rot, txt, (int(posicao[0]),int(posicao[1])-5), fonte, 1.5, cor, 0, cv2.LINE_AA)
                                #self.texto.append({'x':"", 'y':"",'x1':round(linha2[0], 3), 'y1':round(linha2[1], 3),'x2':round(linha2[2], 3), 'y2':round(linha2[3], 3), 'codigo': cont_tm ,'binario':binario,'angulo':"",'tipo':"TM"})
                                #self.store.put('tito', name='Mathieu', age=30)
                                self.contador=self.contador+1
                                self.milliseconds= int(time() * 1000)
                                t1_stop = process_time()
                                last_detect= t1_stop-self.t1_start
                                self.last_detect=int(last_detect*1000)
                                self.t1_start=t1_stop
                                self.store.put(str(self.contador),x="", y="",x1=round(linha2[0], 3), y1=round(linha2[1], 3),x2=round(linha2[2], 3), y2=round(linha2[3], 3), codigo=cont_tm ,binario=binario,angulo="",tipo="TM",time=str(self.milliseconds),duracao=str(self.last_detect ))
                                rgba=rot

        
            
##############################################################################################
        # self.escreve_id(rgba, str(10),(200,200) )
        # #self.escreve_id(rgba, str(1000),(200,200) )
        
       #V=self.Detectect(filtro)
        #rgba   = cv2.cvtColor(rgba, cv2.COLOR_GRAY2RGBA)
        #V=self.Detectect(filtro)
       # if len(V) > 0:
        #  self.escreve_id(rgba, str(10000),(200,200) )
        #P1,P2,ID=self.detect_lowScale(rgba,rgba)
        #if P1:
        #texto=str(len(contours))
        #posicao=(100,100)
        #cor=(0,0,220,255)
        #fonte = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(rgba, texto, (int(posicao[0]),int(posicao[1])-5), fonte, 1.5, cor, 0, cv2.LINE_AA)
  

         # cv2.rectangle(rgba, (int(P2[0]),(P2[1])),(int(P1[0]),int(P1[1])) ,(0,255,0,255), 4)#azul
         # self.escreve_id(rgba, str(ID),P1 )
        #cv2.rectangle(rgba, (20,50),(100,80),(0,255,0,255), 4)#azul
        #rgba = cv2.flip(rgba,1)
        # perms = [Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE]
    
        # if  self.check_permissions(perms)!= True:
        #     request_permissions(perms)    # get android permissions     
        #     exit()                        # app has to be restarted; permissions will work on 2nd start
            
        # try:
        #     #Logger.info('Got requested permissions') 
               
        #     field_names = ['x', 'y','x1', 'y1', 'x2','y2','codigo','binario','angulo','tipo']    
            
        #     fname = os.path.join( primary_external_storage_path(),'test_detect_30_08_2023.csv')
        #     #self.log('writing to: %s' %fname)
        #     self.escreve_id(rgba, 'writing to: %s' %fname,(100,100) )
        #     with open(fname, 'w',encoding='UTF8', newline='') as csvfile:        # write testfile
        #         writer = csv.DictWriter(csvfile, fieldnames=field_names)
        #         writer.writeheader()
        #         writer.writerows(self.texto)
            

        #     #with open(self.path+'test_detect_25_08_2023.csv', 'w',encoding='UTF8', newline='') as csvfile:
    
            
        
        # except:
        #     #self.log('could not write to external storage ... missing permissions ?')    
        #     self.escreve_id(rgba, 'could not write to external storage ... missing permissions ?',(100,200) )
        # # field_names = ['x', 'y','x1', 'y1', 'x2','y2','codigo','binario','angulo','tipo']    
        # # with open(self.path+'test_detect_25_08_2023.csv', 'w',encoding='UTF8', newline='') as csvfile:
    
        # #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
        # #     writer.writeheader()
        # #     writer.writerows(self.texto)
        
        pixels = rgba.tostring()

        self.make_thread_safe(pixels, image_size) 

    @mainthread
    def make_thread_safe(self, pixels, size):
        if not self.analyzed_texture or\
           self.analyzed_texture.size[0] != size[0] or\
           self.analyzed_texture.size[1] != size[1]:
            self.analyzed_texture = Texture.create(size=size, colorfmt='rgba')
            self.analyzed_texture.flip_vertical()
        if self.camera_connected:
            self.analyzed_texture.blit_buffer(pixels, colorfmt='rgba') 
        else:
            # Clear local state so no thread related ghosts on re-connect
            self.analyzed_texture = None
            
    ################################
    # Annotate Screen - on UI Thread
    ################################


    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        # texture : preview Texture
        # size    : preview Texture size (w,h)
        # pos     : location of Texture in Preview Widget (letterbox)
        # Add the analyzed image
        if self.analyzed_texture:
            Color(1,1,1,1)
            Rectangle(texture= self.analyzed_texture,
                      size = tex_size, pos = tex_pos)

    def log(self,msg):
      Logger.info(msg)


    def check_permissions(self,perms):
          for perm in perms:
              if check_permission(perm) != True:
                  return False
          return True

    # def testwrite(self):
    
    #   testfile = "Deus, e, pai"                 # file with 4 bytes
      
      
    #   try:
    #       Logger.info('Got requested permissions')    
          
    #       #fname = os.path.join( primary_external_storage_path(),'testfile')
    #       #self.log('writing to: %s' %fname)
    #       fname=primary_external_storage_path()+"/Download/test.csv"
    #       with open(fname, 'w') as f:        # write testfile
    #           f.write(testfile)
    #       #return fname
      
    #   except:
    #       self.log('could not write to external storage ... missing permissions ?')  

          
    def loadJson(self):
      json_path='metadados.json' 
        
      data=""
      #print(self.path)
      
      f = open (json_path, "r")
      data = json.loads(f.read())
      f.close()
      return data


    def carregaNovaPlanta(self):
        url_planta='sala1.png' 
        img = cv2.imread(url_planta, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        data=self.loadJson()
        
        dim = (w, h)
        # resize image
        img2=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.aspect_metros_x =   w/data[0]['largura'];
        #print(w)
        #aspect_metros_y =   h/data[0]['altura'];
        #print(h)
        return  cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

    # def distorcer_imagem(self,img):
    #     with np.load('calib.npz') as calibData:
    #       mtx, dist, rvecs, tvecs = [calibData[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    #     h, w = img.shape[:2]

    #     newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    #     mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMtx, (w,h), 5)
    #     dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


    #     return dst2

    def lerTMToDraw(self,identificador):
        x1=0
        y1=0
        x2=0
        y2=0
        data=self.loadJson()
        linha=[]
        cont_tm=0
        for i in data:
            if i['idtm']== str(identificador):
            
    #             print(i['x1'])
    #             print(i['y1'])
    #             print(i['x2'])
    #             print(i['y2'])
                x1=i['x1']
                y1=i['y1']
                x2=i['x2']
                y2=i['y2']
                cont_tm=i['n_tm']
                linha.append(x1*self.aspect_metros_x)
                linha.append(y1*self.aspect_metros_x)
                linha.append(x2*self.aspect_metros_x)
                linha.append(y2*self.aspect_metros_x)
                break

        return cont_tm,linha

    def isCM(self,identificador):
    
        isCM=False
        data=self.loadJson()
        #aspect_metros = data[0]['largura'] / 800;
        for i in data:
            if i['idcm'] == identificador:
                isCM=True
           
        return isCM

    def lerCMToDecimal(self,identificador):
        decimal=0
        y=0
        #isCM=0
        pontos=[]
        data=self.loadJson()
        #aspect_metros = data[0]['largura'] / 800;
        for i in data:
            if i['idcm']== identificador:
               # isCM=1
                #print(i['idcm_decimal'])
            
                decimal=i['idcm_decimal']
            
        return decimal

    def lerCMToChecsum(self,identificador):
        checksum=0
        #y=0
        #isCM=0
        #pontos=[]
        data=self.loadJson()
        #aspect_metros = data[0]['largura'] / 800;
        for i in data:
            if i['idcm']== str(identificador):
                #isCM=1
                #print(str(i['checksum']) )
            
                checksum=i['checksum']
            
        return checksum

    def lerCMToDraw(self,identificador):
        x=0
        y=0
        #isCM=0
        pontos=[]
        data=self.loadJson()
        #aspect_metros = data[0]['largura'] / 800;
        for i in data:
            if i['idcm_decimal']== identificador:
                #isCM=1
                #print(i['x'])
                #print(i['y'])
                x=i['x']
                y=i['y']
                pontos.append(x*self.aspect_metros_x)
                pontos.append(y*self.aspect_metros_x)
            
        #print(pontos)
        return pontos

    def escreve_local(self,img, texto, posicao, cor=(0,0,255)):
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, texto, (int(posicao[0]+10),int(posicao[1]+30)), fonte, 1.3, cor, 2, cv2.LINE_AA)

   # def validaChecksum(bits):
    #    h = blake2b(digest_size=1)
   #     h.update(bits)
    #    x=h.hexdigest()
        #print(x)
   #     hex_string = x
# Convert the hexadecimal string to an integer using the base 16.
    #    hex_integer = int(hex_string, 16)
# Convert the integer to binary using the bin() function.
    
     #   binary_string = bin(hex_integer)
     #   saida=""
    #    if len(binary_string)==8:
    #        saida="00"+binary_string[2:]
    #    elif len(binary_string)==7:
     #       saida="000"+binary_string[2:]
     #   elif len(binary_string)==6:
        
     #       saida="0000"+binary_string[2:]
        
    #    elif len(binary_string)==5:
        
      #      saida="00000"+binary_string[2:]
      #  elif len(binary_string)==10:
     #       saida=""+binary_string[2:]
      #  elif len(binary_string)==9:
     #       saida="0"+binary_string[2:]
        
        #print(binary_string)    
      #  return saida

    # def showMap( self,pontos,decimal):
    #     global resized_planta

    #     x=pontos[0]
    #     y=pontos[1]
    #     resized_planta=self.carregaNovaPlanta()
    #     image_out = cv2.circle(resized_planta, (int(x),int(y)), 1, (0, 255, 0), 15)#azul
    #     self.escreve_local(image_out, ""+str(decimal),pontos )
    #     #plt.figure()
    #     #plt.xticks([])              # set no ticks on x-axis
    #    # plt.yticks([])
    #    # plt.imshow(image_out)
    #     #plt.title('Indoor localization System escala menor')
    #     #plt.show()
    #     #cv2.imshow("rot@22", image_out)
    #     #image_id = "picture " + str(self.count) + ".jpg"
    #     #cv2.imwrite("Camera_Pictures/" + image_id, self.frame)
    #     #self.count += 1
    #     #cv2.imwrite(url_planta, image_out)
        
    #    # print(url_planta)
    #     #self.root.ids.img.source = url_planta
    #     #print(path_planta)
    #     image_id = path_planta+"/" +"picture " + str(decimal) + ".png"
       
        
    #     cv2.imwrite(image_id, image_out)
    #     global url
    #     url=image_id
    #     #self.get_root_window.ids.img.source = image_id
    #    # print(image_id)


    
#     def showLineMap(self,pontos,bits):
#         global resized_planta
#     #Img = np.zeros((400, 800, 3), dtype='uint8')
#         x1=pontos[0]
#         y1=pontos[1]
#         x2=pontos[2]
#         y2=pontos[3]
#         start_point = (int(x1), int(y1))
# # End coordinate, here (450, 450). It represents the bottom right corner of the image according to resolution
#         end_point = (int(x2), int(y2))
# # White color in BGR
#         color = (0, 255, 0)
# # Line thickness of 9 px
#         thickness = 2
#     #image_out = cv2.circle(resized_planta, (int(x),int(y)), 1, (0, 255, 0), 15)#azul
#     #Img = cv2.circle(Img, (int(x2),int(y2)), 1, (255, 255, 0), 3)#azul
# #     image_out = cv2.line(resized_planta, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
#         Img = cv2.line(resized_planta, start_point, end_point, color, thickness)
    
#         self.escreve_local(Img, ""+str(bits),pontos )
#         #plt.figure()
#         #plt.xticks([])              # set no ticks on x-axis
#         #plt.yticks([])
#         #plt.imshow(Img)
#         #plt.title('Indoor localization System escla maior')
#         #plt.show()
#         self.root.ids.img =Img

    def getLine(self,img):
        Array_linhas=[]
    
        h, w = img.shape[:2]
        rot=np.zeros((h,w))
    
        #cv2.imshow("rot@22", img) 
        gray   = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    #__________________________________________________
    #plt.figure()
   # plt.imshow(gray,cmap='gray')
   # plt.title('Imagem cinza')
    #plt.show()
        dst = cv2.Canny(gray, 255, 255, apertureSize =3)
#     dst = cv2.Canny(gray, 230, 255, apertureSize =3)
        #cv2.imshow("rot@22222", dst) 
    
    #__________________________________________________
   # plt.figure()
   # plt.imshow(dst,cmap='gray')
   # plt.title('Imagem canny')
   # plt.show()
    
#     lines_1 = cv2.HoughLines(dst, 1, np.pi / 180, 240)
        lines_1 = cv2.HoughLines(dst, 1, np.pi / 180, 240)
    #print(lines_1) 
        Mt=[]
    
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)  
    #rotacionar imagem 
        if lines_1 is not None:
        
            for l in lines_1:
                lines_list=[]
                rho = l[0][0]
                theta = l[0][1]
                a = math.cos(theta)
                b = math.sin(theta) 
                x0 = a * rho
                y0 = b * rho       
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                # rotate our image by Theta degrees around the center of the image
                M = cv2.getRotationMatrix2D((cX, cY), math.degrees(theta) -90, 1.0)
            
                lines_list.append([pt1[0],pt2[0] ]) #X
                lines_list.append([pt1[1],pt2[1] ] ) #Y
                lines_list.append([1,1 ] )
            
                Array_linhas.append({'M':M,'lines_list':lines_list})

            return Array_linhas
        else:
            return None
   

 #função que converte a cadeia de bits para modulos de bits

    def bit_to_bars(self,string):

        bars = []
      
        current_length = 1
      
        lenght=0
        if len(string)>0:
            for i in range(len(string)-1):
                if string[i] == string[i+1]:
                    current_length = current_length + 1
                else:
                    bars.append(current_length * str(string[i]))
                    current_length = 1
          
        return bars

    def remove_quieteZone(self,b):
        barra_q=b
        valida=0
        inicio=0
        fim=0
    
        cont=0
        for i in range(0,len(barra_q)-9):
            
            if  (  abs(len(barra_q[i])-len(barra_q[i+2]))< 2 and abs(len(barra_q[i+1])-len(barra_q[i+3]))< 2  and abs(len(barra_q[i+4])-len(barra_q[i+6]))< 2 and len(barra_q[i+9]) > len(barra_q[i+8]) ):
                inicio=i
                break
                #barra_q[i][0]=="1" and barra_q[i+2][0]=="1" and          
   
        for j in range(inicio-1):
            barra_q.pop(0)         
    ##inversa de traz para frente and barra_q[i+2] == "1" and barra_q[i+4] == "1" and barra_q[i+6] == "1"
        cont=0
    
        for i in range(len(barra_q)-1,0,-1):
            if len(barra_q) >2:
            
                if  ( abs(len(barra_q[i])-len(barra_q[i-2]))< 2 and abs(len(barra_q[i-1])-len(barra_q[i-3]))< 2  and abs(len(barra_q[i-4])-len(barra_q[i-6]))< 2 and len(barra_q[i-7]) < len(barra_q[i-8]) ):
                    fim=i
                    break
                             
    
        for j in range(len(barra_q)-1,fim+1,-1):
            barra_q.pop(len(barra_q)-1)                
    
        return barra_q 


    def remove_barras(self,barra):
        new_bar=barra  
        if len(new_bar) > 60:
            for j in range(8):
                new_bar.pop(0)         
        ##inversa
        
            for j in range(7):
                if len(new_bar) > 7:
                    new_bar.pop(len(new_bar)-1)                
        #remove marcador
        
    
        return new_bar

#função que remove os marcadores que antecedem a região de encoder (quadrados )para extração das caracteristicas do marcador em escala maior
    def remove_quadrado(self,barra):
        bar=barra
        new=[]
        new2=[]
        maior=0
        if len(bar) > 0:
            for k in range(len(bar)-2):
                if abs(len(bar[k])-len(bar[k+1]) ) > 5 and abs(len(bar[k+1])-len(bar[k+2]) ) > 5:
                    new.append(bar[k+1])
                if len(bar[k]) > maior:
                    maior= len(bar[k])   
        
            for i in range(len(new)) :
                if abs(len(new[i])- maior) < 10:
                    new2.append(new[i])                        
      
        return new2
#função que codificas as cadeias de bits em valores unitarios 111111100000001111111 =101(simplificação)
    def encode_bars(self,array_bar):
    
        array=array_bar
        s = ""
        for value in array:
            if value[0]=="1":
                s = s + "0"
            if value[0]=="0":
                s = s + "1"
      
        return s

#função que converte as linhas da imahgem em array de bits 0 e 1 
    def array_to_string(self, array1):
        array=array1
        s = ""
        for value in array:
            s = s + str(value)
    # print("Data string: " + s)
        return s

#função que valida as os codigos que tem maior incidencia nas linhas
    def valida_detect(self,array):
   
        bi_t=""
        saida=""
        tp=0
        if len(array) >0:
            for k in range(0,len(array)):
                contador=0
                binario=array[k]        
                for i in range(0,len(array)):
                    if array[i]==binario:
                        contador=contador+1
                   # print("contador de repeticao : ",contador)            
                    if tp<contador:
                        tp=contador
                        bi_t=binario
            if abs(tp/len(array))>=0.4:
                saida=bi_t
        return saida 
#escreve tento na imagemde detecção maior
    def escreve(self,img, texto, cor=(0,0,220)):
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, texto, (250,250), fonte, 10.5, cor, 0, cv2.LINE_AA)


# função de detecção em escala maior
    def detect_HighScale(self,img):
        bit=""
        binario=""
        sucesso=False
        sucesso2=False
        linha2=[]
        Arr=self.getLine(img)
        (h, w) = img.shape[:2]
        linha=[]
        if not Arr is None:
            for lin in Arr: 
                #print(lin['M'],lin['lines_list']) 
                linha=np.dot(lin['M'],lin['lines_list']).astype(np.int32)
            
                img_rot= cv2.warpAffine(img, lin['M'], (w, h))
            
                rot=img_rot.copy()
                encode=[]
                if len(linha) > 0:
                    if rot.size>0:
                        x1=linha[0][0]-w
                        y1=linha[1][0]
                        x2=linha[0][1]+w
                        y2=linha[1][1]
                    
                  
                    
                        #cv2.line(rot,(linha[0][0],linha[1][0]),(x2,y2),(255,255,0),5)
                    
                    #--------
                   # plt.figure()
                   # plt.imshow(rot)
                   # plt.title('Imagem Rotacionada')
                   # plt.show()
                        
                        crop_img=img_rot[y1-100:y2+10,x1: x2]
                        h1, w1 = crop_img.shape[:2]
                    
                   
                    #--------
#                     plt.figure()
#                     plt.imshow(crop_img)
#                     plt.title('Imagem crop')
#                     plt.show()
                   
                        if h1 > 0 and w1 > 0:
                    
                        #temp=FiltragemInicial(crop_img)
                            temp = cv2.cvtColor(crop_img, cv2.COLOR_RGBA2GRAY)
                        #gray   = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
                        
                            suave = cv2.GaussianBlur(temp, (7, 7), 0) # aplica blur
                            ret,thresh = cv2.threshold(suave,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


                        #temp= cv2.adaptiveThreshold(temp, 255,	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, 1)
                        
                            thresh = cv2.bitwise_not(thresh)
                            #cv2.imshow("Sourc", rot)
                        
                       #_____________________________
                       # plt.figure()
                        #plt.imshow(thresh,cmap="gray")
                        #plt.title('Imagem binaria')
                        #plt.show()
                        #_____________________________
                        
                            for l in range(0, h1-1):
                            
                                lines= thresh[l]
                            
                                for i in range(len(lines)):
                                    if lines[i] == 255:
                                        lines[i] = 1
                            
                                string=self.array_to_string(lines)
                                #print(l,"cadeia__ ",string)
                                bar=self.bit_to_bars(string)
                            #print(l,"bar",bar)
                                data_string=self.remove_quieteZone(bar)
                                
                                ## varrer um por um e contar os padroes 
                                
                           
                               # print(l,"limpa e valida",data_string)
                                sbarras=self.remove_barras(data_string)
                            #print(l,"sem barras ",sbarras)
                                squadrado=self.remove_quadrado(sbarras)
                               # print(l,"sem quuadro ",squadrado)
                                bits=self.encode_bars(squadrado)
                            
                                if len(bits)>2:
                                    encode.append(bits)
                                
                                                  
                                #print("akiiiiiiiiiiii",bits)
                    bit=self.valida_detect(encode)
                   # print(bit)
                    if(len(bit) > 2):
                        linha2=self.lerTMToDraw(bit)
                        if not linha2:
                            break               #print("Detectou em escal maior : ",bit)
                            #sucesso=True
                            
                        else:
                            binario=str(bit)
                            #print(binario)
                           # self.escreve(rot, str(binario) ) 
                           
                            #cop=cv2.cvtColor(rot, cv2.COLOR_BGR2RGB) 
                            #######cv2.imshow("Imagem detect escala maior",cop)
                   # plt.figure()
                   # plt.title('Imagem detect')
                   # plt.imshow(cop)
                   # plt.show()
                
                            #self.showLineMap(linha,bit)
   
        return binario

    def FiltragemInicial(self,img):
        # obj3=cv2.medianBlur(img,3)
        # B,G,R,_ =cv2.split(obj3)
 
    
        # v1= np.absolute(R.astype(float)-B.astype(float)) < 30
    
        # v2= np.absolute(G.astype(float)-B.astype(float)) < 30
    
        # v3= np.absolute(R.astype(float)-G.astype(float)) < 30
    
        # obj2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
        # v6= obj2 < 60
        # v7= obj2 > 155 
    
        # v4=np.logical_and(v1,v2)
        # v5=np.logical_and(v3,v4)
    
    
        # v8=np.logical_and(v5,v6)
    
        # v9=np.logical_and(v5,v7)
       
        # obj2[v8]=[0]
        # obj2[v9]=[255]
        #cv2.imshow("warp temp 1", obj2)
        gray   = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img   = cv2.GaussianBlur(gray, (3,3), 0)
        return img

#função que faz a detecção dos contornos quadrados que delimitam a região de encoder    
    def Detectect(self,img):
      V=[]
      sucesso2=False  
      img = cv2.equalizeHist(img)
      obj= cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 201, 1)
      contours, hierarchy= cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      cont=0
      for i in range(0,len(contours) ):
        if cv2.arcLength(contours[i],True)> 10:
          if hierarchy[0][i][2] >= 0 and hierarchy[0][i][3] == -1:
            rect = cv2.minAreaRect(contours[i])
            aspect=rect[1][0]/rect[1][1]
            epsilon = 0.1*cv2.arcLength(contours[i],True)
            approx = cv2.approxPolyDP(contours[i],epsilon,True)
            if aspect > 0.6 and approx.shape[0]==4:
              j=hierarchy[0][i][2]
              if cv2.arcLength(contours[j],True)> 10:
                if hierarchy[0][j][2] >= 0:
                  rect_j = cv2.minAreaRect(contours[j])
                  aspect_j=rect_j[1][0]/rect_j[1][1]
                  epsilon_j = 0.1*cv2.arcLength(contours[j],True)
                  approx_j = cv2.approxPolyDP(contours[j],epsilon_j,True)
                  if aspect_j > 0.6 and approx_j.shape[0]>=3:
                    k=hierarchy[0][j][2]
                    if cv2.arcLength(contours[k],True)> 10:
                      if hierarchy[0][k][3] >= 0:
                        rect_k = cv2.minAreaRect(contours[k])
                        if rect_k[1][1]>0:
                          aspect_k=rect_k[1][0]/rect_k[1][1]
                          epsilon_k = 0.1*cv2.arcLength(contours[k],True)
                          approx_k = cv2.approxPolyDP(contours[k],epsilon_k,True)
                          if aspect_k > 0.6 and approx_k.shape[0]>=2:
                            cont=cont +1
                            retangulo = cv2.minAreaRect(contours[i])
                            box = cv2.boxPoints(retangulo)
                            V.append(box)
 
                                                   # p1=[0,0]
                                                   # p2=[0,0]
                                                   # p3=[0,0]
                                                   # p4=[0,0]
                                                    
                                                   # if len(box):
                                                        
                                                        #p[0] = p[0]// round((400 / int(self.root.ids.camera.height)), 2)
                                                        #p[1] = self.root.ids.camera.height - p[1]
                                                        #p[3] = p[3] * -1
                                                        
                                                        
                                                        #p1[0]=int(box[0][0] )
                                                        #p1[1]=int(self.root.ids.camera.height - box[0][1]+120)
                                                        #self.root.ids.camera.p1=p1
                                                        #p2[0]=int(box[1][0])
                                                       # p2[1]=int(self.root.ids.camera.height - box[1][1])+120
                                                       # self.root.ids.camera.p2=p2
                                                       # p3[0]=int(box[2][0])
                                                       # p3[1]=int(self.root.ids.camera.height - box[2][1])+120
                                                       # self.root.ids.camera.p3=p3
                                                       # p4[0]=int(box[3][0])
                                                       # p4[1]=int(self.root.ids.camera.height - box[3][1])+120
                                                       # self.root.ids.camera.p4=p4
                                                       # p=[p1[0],p1[1],p3[0],p3[1] ]
                                                        #self.root.ids.camera.face=p
            
                                                    #else:
                                                       # self.root.ids.camera.face=[0,0,0,0]
                                                       # self.root.ids.camera.p1 = [0, 0]
                                                      #  self.root.ids.camera.p2 = [0, 0]
                                                      #  self.root.ids.camera.p3 = [0, 0]
                                                      #  self.root.ids.camera.p4 = [0, 0]
                                                    #image_out = cv2.circle(frame2, (int(box[0][0]),int(box[0][1])), 1, (255, 0, 0), 15)#azul
                                                   # image_out = cv2.circle(frame2, (int(box[1][0]),int(box[1][1])) , 1, (255, 0, 255), 15)#rosa
                                                   # image_out = cv2.circle(frame2, (int(box[2][0]),int(box[2][1])) , 1, (0, 255, 0), 15)#verde
                                                   # image_out = cv2.circle(frame2, (int(box[3][0]),int(box[3][1]) ) , 1, (0, 0, 255), 15)#vermelho
                                                
                                                   # cop=cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)  
                                                   # cv2.imshow("Imagem com pontos",cop)
                                                #plt.figure()
                                                #plt.title('Imagem pontoooooooooo neto')
                                                #plt.imshow(cop)
                                                #plt.show()
                                                
    
      return V

# função que realiza a extração da informação da região do encoder em escala menor 
    
    def Extract(self,final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame):
        saida2=""
        B=[]
        tag=False
        warped_img=self.Warp(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
    #######
    #warped_img=Warp(final_bottom_left,final_bottom_right,final_top_left,final_top_right,frame)

        temp=self.FiltragemInicial(warped_img.copy())
    #temp= cv2.cvtColor(temp.copy(), cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(temp, (13, 13), 0) # aplica blur
        ret,th = cv2.threshold(suave,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #obj_bin= cv2.adaptiveThreshold(temp, 200,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        imagem_cinza=temp
        #print(th)
    #plt.figure()
    #plt.title('Imagem bin ')
   # plt.imshow(th, cmap="gray")
   # plt.show() 
    
    #cv2.imshow("warp temp 1", temp)
        #print(warped_img.shape)
        A=np.zeros((7,16))
        size=832/32
#     size=1000/32
        for a in range (0,16 ):
            for b in range (0,7 ):
                col=(a+8)*size      
                lin=b*size
                cropped_image = th[int(lin):int(lin+size), int(col): int(col +size)]
            
            
                if cropped_image[9,9] == 255 : # pixel maiores que 127 viram 1
                    A[b,a]=1
                    cv2.rectangle(warped_img, (int(col),int(lin)),(  int(col+ size),int(lin+size)) ,(255, 0, 0), 1)                                        
            #cv2.imshow("tem", warped_img)
           
   # plt.figure()
   # plt.title('Imagem bin warp crop')
   # plt.imshow(warped_img, cmap="gray")
   # plt.show() 
    
        f1=True
        f2=True
    
        for a in range (0,16 ):
            if A[6,a]==1:
                f1=False
        for b in range (0, 7):
            if A[b,0]==1:
                f2=False
        if A[0,14]==A[0,15]:
            f2=False
        if f1 and f2:
            B = A
   
    
        #print(B)
        bits=""
        Str=""
        saida=""
        checksum=""
        saida_check=""
        for l in B:
            Str=Str+str(l).replace("[", "").replace("]", "").replace(".", "").replace(" ", "")
        #print("aquiii nnn ", Str )
#     Str=Str[1:-15]
        #print("aquiii 1 ", Str )
        if len(Str) >100:
            checksum=checksum+str(Str[18] )
            checksum=checksum+str(Str[21] )
            checksum=checksum+str(Str[24] )
            checksum=checksum+str(Str[29] )
            checksum=checksum+str(Str[66] )
            checksum=checksum+str(Str[69] )
            checksum=checksum+str(Str[72] )
            checksum=checksum+str(Str[77] )
        #print("check  ",checksum)
            for i in range(0, len(Str) ):
                if i==0: #bit orientação
                    continue
                elif i==14: #bit orientação
                    continue
                elif i==15: #bit orientação
                    continue
                elif i==16:
                    continue
                elif i==18:
                    continue
                elif i==21:
                    continue    
                elif i==24:
                    continue
                elif i==29:
                    continue    
                elif i==30:
                    continue
                elif i==31:
                    continue
                elif i==32:
                    continue
                elif i==48:
                    continue
                elif i==64:
                    continue
                elif i==66:
                    continue
                elif i==69:
                    continue
                elif i==72:
                    continue
                elif i==77:
                    continue    
                elif i==80:
                    continue
                elif i==96:
                    continue
                elif i==97:
                    continue
                elif i==98:
                    continue
                elif i==99:
                    continue
                elif i==100:
                    continue
                elif i==101:
                    continue
                elif i==102:
                    continue
                elif i==103:
                    continue
                elif i==104:
                    continue
                elif i==105:
                    continue
                elif i==106:
                    continue
                elif i==107:
                    continue
                elif i==108:
                    continue
                elif i==109:
                    continue
                elif i==110:
                    continue
                elif i==111:
                    continue

                else:
                    bits=bits+str(Str[i] )
       # print("aquiii 2 ", bits )
   
        for i in range(0, len(bits)):
            if bits[i]=="1":
                saida+="0"
                
            elif bits[i]=="0":
                saida+="1"
        for i in range(0, len(checksum)):
            if checksum[i]=="1":
                saida_check+="0"
                
            elif checksum[i]=="0":
                saida_check+="1"
    
        #print("aquiii 3", saida)
        #print("aquiii ",self.lerCMToChecsum(saida) )
        #print("check  ",saida_check)
    
       
                  
        if self.lerCMToChecsum(saida)== saida_check:
            #print("check OK!")
            return saida
        else:
            return ""

    def draw_boundBox(self,final_top_left,final_bottom_right,copy):
        #box = np.uint32([0,0,0,0])
       # box=[0,0,0,0]
        copy = cv2.rectangle(copy, (int(final_bottom_right[0]),(final_bottom_right[1])),(int(final_top_left[0]),int(final_top_left[1])) ,(0,255,0,255), 4)#azul
       # h1, w1 = copy.shape[:2]
        #h2, w2 = self.root.ids.camera.resolution
        #largura=self.root.ids.camera.largura
       # altura=self.root.ids.camera.altura
       # asp_l=largura/w1
       # asp_a=altura/h1
       # print("imagem: ")
       # print(copy.shape[:2])
        #print("camera: ")
        #print(largura)
       # print(altura)
        #global url_new
        #url_new='C:/Users/Benedito Neto/Downloads/planta2.png'

       # box[0]=int(final_bottom_right[0])
       # box[1]=int(self.root.ids.camera.height -final_bottom_right[1] )+120
       # box[2]=int(final_top_left[0])
       # box[3]=int(self.root.ids.camera.height - final_top_left[1])+120
        #self.root.ids.camera.pose = [box[0],box[1]]
        #self.root.ids.camera.pose = [int(final_bottom_right[0]),int(final_bottom_right[1])]
        #self.root.ids.camera.pose = [int(final_top_left[0]),int(final_top_left[1])]
       # self.root.ids.camera.pose = [box[0],box[1]]
        #print(box[0])
        #print(box[1])
        #print(box[2])
        #print(box[3])
        #print(self.root.ids.camera.resolution)
        #print(h2)
       # if len(final_top_left):
       #     self.root.ids.camera.face = box
       # else:
         #   self.root.ids.camera.face = [0,0,0,0]
            
        return copy

# função que faz a transformação da perspectiva da imagem da região de encoder ficar plana para extração das caracteristicas     

    def Warp(self,final_top_left,final_top_right,final_bottom_left,final_bottom_right,img):
    
        input_pts = np.float32([[final_top_left],[final_top_right],[final_bottom_left],[final_bottom_right] ])
    
#     output_pts = np.float32([[0,0],[1000,0],[0,219],[1000,219]]) # matriz homografica
#     output_pts = np.float32([[0,0],[832,0],[0,183],[832,183]]) # matriz homografica
        output_pts = np.float32([[0,0],[832,0],[0,182],[832,182]]) # matriz homografica
   
   
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
 
    # Apply the perspective transformation to the image
#     warped_img = cv2.warpPerspective(img,M,(1000, 219))
        warped_img = cv2.warpPerspective(img,M,(832, 182))
    #   cv2.imshow("teste",warped_img)
      
   # warped_img2=draw_points(final_top_left,final_top_right,final_bottom_left,final_bottom_right,warped_img)
                    
   # plt.figure()
   # plt.title('Imagem points Warp2')
   # plt.imshow(warped_img,cmap="gray")
   # plt.show()
    
        return warped_img 


#escreve tento na imagem de detecção menor
    def escreve_id(self,img, texto, posicao, cor=(0,0,220,255)):
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, texto, (int(posicao[0]),int(posicao[1])-5), fonte, 0.5, cor, 0, cv2.LINE_AA)

    def rotate_detect90(self,V,frame):
        bits=""
        pose1=[]
        pose2=[]
        result=[]
        for i in range(0,len(V)-1 ):
            if len(V[i])==4:
                                     
                p1,p2,p3,p4=V[i]
                
                p5,p6,p7,p8=V[i+1]

                final_top_right=(p3[0],p3[1])#azul0
                final_top_left=(p6[0],p6[1])#vermelho
                final_bottom_right=(p4[0],p4[1])#rosa0
                final_bottom_left=(p5[0],p5[1])#verde
                bits=self.Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                pose1=(int(p1[0]),int(p1[1]))
                pose2=(int(p7[0]),int(p7[1]))
                     
                if self.isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                    
                else:
                    result=[]
                    
        return result

    def rotate_detect45(self,V,frame):
        bits=""
        pose1=[]
        pose2=[]
        result=[]
        dados=[]
        for i in range(0,len(V)-1 ):
            if len(V[i])==4:
                p1,p2,p3,p4=V[i]
                p5,p6,p7,p8=V[i+1]

                final_top_left=(p1[0],p1[1])#azul
                final_bottom_left=(p4[0],p4[1])#verme
                final_top_right=(p6[0],p6[1])#rosa
                final_bottom_right=(p7[0],p7[1])#verde
                bits=self.Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                pose2=(int(p3[0]),int(p3[1]))
                pose1=(int(p5[0]),int(p5[1]))
                    
                    
                if self.isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                    
                else:
                    result=[]
                    
        return result

    def rotate_detect_0(self,V,frame):
        bits=""
        pose1=[]
        pose2=[]
        result=[]
        dados=[]
        for i in range(0,len(V)-1 ):
            if len(V[i])==4:
                p1,p2,p3,p4=V[i+1]
                p5,p6,p7,p8=V[i]

                final_top_left=(p1[0],p1[1])#azul
                final_bottom_left=(p4[0],p4[1])#verme
                final_top_right=(p6[0],p6[1])#rosa
                final_bottom_right=(p7[0],p7[1])#verde
                bits=self.Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                pose1=(int(p3[0]),int(p3[1]))
                pose2=(int(p5[0]),int(p5[1]))
                    
                    
                if self.isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                    
                else:
                    result=[]
                    
        return result

    def rotate_detect180(self,V,frame):
        bits=""
        pose1=[]
        pose2=[]
        result=[]
        for i in range(0,len(V)-1 ):
            if len(V[i])==4:
                p1,p2,p3,p4=V[i]
                
                p5,p6,p7,p8=V[i+1]

                final_bottom_right=(p1[0],p1[1])#azul
                final_top_right=(p4[0],p4[1])#verme
                final_bottom_left=(p6[0],p6[1])#rosa
                final_top_left =(p7[0],p7[1])#verde
                bits=self.Extract(final_top_left,final_top_right,final_bottom_left,final_bottom_right,frame)
                pose1=(int(p3[0]),int(p3[1]))
                pose2=(int(p5[0]),int(p5[1]))
                    
                if self.isCM(str(bits)):
                    result.append(pose1)
                    result.append(pose2)
                    result.append(bits)
                    
                else:
                    result=[]
                    
        return result        
                        
#função de detecção em escala menor
    def detect_lowScale(self,img_detect,copy):
        sucesso=False
        id1=None
        pontos=[]
        V=[]
        V_info=[]
        p1=0
        p2=0
        D=[]
       # algoritmo = str("escala menor")
        img2=self.FiltragemInicial(img_detect)
        V=self.Detectect(img2)
    
        V_info=self.rotate_detect45(V,copy)
        #print((V_info) )
        if not V_info:
            #print("esta vazia lista!")
            sucesso=False
        else:
            for i in range(0,len(V_info),3 ):
                #print(len(V_info[i]))
                #print(" passou!  ",V_info[i])
                p1=V_info[i]
                p2=V_info[i+1]
                bits=V_info[i+2]
                #print("bits_________",bits)    
                id1=self.lerCMToDecimal(str(bits) )
                
                #copy=self.draw_boundBox(p1,p2,copy)
                #self.escreve_id(copy, ""+str(id1),p1 )
                #cop=cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)    
                
               # plt.figure()
               # plt.xticks([])              # set no ticks on x-axis
               # plt.yticks([])
               # plt.title('Imagem teste 45')
               # plt.imshow(cop)
               # plt.show()
                #cv2.imshow("Imagem teste 45",copy)
                #self.root.ids.camera.face=cop
                pontos=self.lerCMToDraw(id1)
                self.contador=self.contador+1
                self.milliseconds= int(time() * 1000)
                t1_stop = process_time()
                last_detect= t1_stop-self.t1_start
                self.last_detect=int(last_detect*1000)
                self.t1_start=t1_stop
                self.store.put(str(self.contador),x=round(pontos[0], 3), y=round(pontos[1], 3),x1="", y1="",x2="", y2="", codigo=id1 ,binario=str(bits),angulo="45",tipo="CM",time=str(self.milliseconds),duracao=str(self.last_detect ))
                D.append(p1)
                D.append(p2)
                D.append(id1)
                #self.texto.append({'x':round(pontos[0], 3), 'y':round(pontos[1], 3),'x1':"", 'y1':"",'x2':"", 'y2':"", 'codigo': id1 ,'binario':str(bits),'angulo':"45",'tipo':"CM"})
               
               # self.showMap(pontos,id1)
                sucesso=True
        
        V_info=self.rotate_detect90(V,copy)
        #print((V_info) ) 
        if not V_info:
            #print("esta vazia lista!")
            sucesso=False
        else:
            for i in range(0,len(V_info),3 ):
                #print(len(V_info[i]))
                #print(" passou!  ",V_info[i])
                p1=V_info[i]
                p2=V_info[i+1]
                bits=V_info[i+2]
                #print("bits_________",bits)    
                id1=self.lerCMToDecimal(str(bits) )
               # copy=self.draw_boundBox(p1,p2,copy)
                #self.escreve_id(copy, ""+str(id1),p2 )
                #cop=cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)    
            
               # plt.figure()
               # plt.xticks([])              # set no ticks on x-axis
               # plt.yticks([])
               # plt.title('Imagem teste 90')
                #plt.imshow(cop)
               # plt.show()
                #cv2.imshow("Imagem teste 90",cop)
                pontos=self.lerCMToDraw(id1)
                self.contador=self.contador+1
                self.milliseconds= int(time() * 1000)
                t1_stop = process_time()
                last_detect= t1_stop-self.t1_start
                self.last_detect=int(last_detect*1000)
                self.t1_start=t1_stop
                #self.texto.append({'x':round(pontos[0], 3), 'y':round(pontos[1], 3),'x1':"", 'y1':"",'x2':"", 'y2':"", 'codigo': id1 ,'binario':str(bits),'angulo':"90",'tipo':"CM"})
                self.store.put(str(self.contador),x=round(pontos[0], 3), y=round(pontos[1], 3),x1="", y1="",x2="", y2="", codigo=id1 ,binario=str(bits),angulo="90",tipo="CM",time=str(self.milliseconds),duracao=str(self.last_detect ))
                D.append(p1)
                D.append(p2)
                D.append(id1)
                #self.showMap(pontos,id1)
                #sucesso=True
    
        V_info=self.rotate_detect180(V,copy)
        #print((V_info) )
        if not V_info:
            #print("esta vazia lista!")
            sucesso=False
        else:
            for i in range(0,len(V_info),3 ):
                #print(len(V_info[i]))
                #print(" passou!  ",V_info[i])
                p1=V_info[i]
                p2=V_info[i+1]
                bits=V_info[i+2]
                #print("bits_________",bits)    
                id1=self.lerCMToDecimal(str(bits) )
                #copy=self.draw_boundBox(p1,p2,copy)
                #self.escreve_id(copy, ""+str(id1),p1 )
               # cop=cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)    
            
               # plt.figure()
               # plt.xticks([])              # set no ticks on x-axis
                #plt.yticks([])
                #plt.title('Imagem teste 180')
                #plt.imshow(cop)
                #plt.show()
                #
                #cv2.imshow("Imagem teste 180",cop)
                pontos=self.lerCMToDraw(id1)
                self.contador=self.contador+1
                self.milliseconds= int(time() * 1000)
                t1_stop = process_time()
                last_detect= t1_stop-self.t1_start
                self.last_detect=int(last_detect*1000)
                self.t1_start=t1_stop
                #self.texto.append({'x':round(pontos[0], 3), 'y':round(pontos[1], 3),'x1':"", 'y1':"",'x2':"", 'y2':"", 'codigo': id1 ,'binario':str(bits),'angulo':"180",'tipo':"CM"})
                self.store.put(str(self.contador),x=round(pontos[0], 3), y=round(pontos[1], 3),x1="", y1="",x2="", y2="", codigo=id1 ,binario=str(bits),angulo="180",tipo="CM",time=str(self.milliseconds),duracao=str(self.last_detect ))
                D.append(p1)
                D.append(p2)
                D.append(id1)
                #self.showMap(pontos,id1)
                #sucesso=True
        V_info=self.rotate_detect_0(V,copy)
        #print((V_info) )
        if not V_info:
            #print("esta vazia lista!")
            sucesso=False
        else:
            for i in range(0,len(V_info),3 ):
               # print(len(V_info[i]))
                #print(" passou!  ",V_info[i])
                p1=V_info[i]
                p2=V_info[i+1]
                bits=V_info[i+2]
               # print("bits_________",bits)    
                id1=self.lerCMToDecimal(str(bits) )
                #copy=self.draw_boundBox(p1,p2,copy)
                #self.escreve_id(copy, ""+str(id1),p1 )
                #cop=cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)    
                #plt.xticks([])              # set no ticks on x-axis
                #plt.yticks([])
                #plt.figure()

                #plt.title('Imagem teste 0')
                #plt.imshow(cop)
                #plt.show()
                #cv2.imshow("Imagem teste 0",cop)
                pontos=self.lerCMToDraw(id1)
                #self.texto.append({'x':round(pontos[0], 3), 'y':round(pontos[1], 3),'x1':"", 'y1':"",'x2':"", 'y2':"", 'codigo': id1 ,'binario':str(bits),'angulo':"0",'tipo':"CM"})
                self.contador=self.contador +1
                self.milliseconds= int(time() * 1000)
                t1_stop = process_time()
                last_detect= t1_stop-self.t1_start
                self.last_detect=int(last_detect*1000)
                self.t1_start=t1_stop
                self.store.put(str(self.contador),x=round(pontos[0], 3), y=round(pontos[1], 3),x1="", y1="",x2="", y2="", codigo=id1 ,binario=str(bits),angulo="0",tipo="CM",time=str(self.milliseconds),duracao=str(self.last_detect ))
                D.append(p1)
                D.append(p2)
                D.append(id1)               
               # self.showMap(pontos,id1)
                #sucesso=True
        #self.draw_boundBox(p1,p2,copy)        
        return D

