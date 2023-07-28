import numpy as np

#PRINTING INPUTS

adapthres = 0.0
""" 
shape = ()
size =  1
min:  0.0
mean:  0.0
max:  0.0
dimension 0 """
#******** bpol ********
bpol = np.zeros(shape=(512, 512),dtype=float)
""" <class 'numpy.ndarray'>
>f4
shape =  (512, 512)
size =  262144
min:  0.0
mean:  0.0
max:  0.0
dimension 2 """
#******** btot ********
btot = np.zeros(shape=(512, 512),dtype=float)
""" <class 'numpy.ndarray'>
>f4
shape =  (512, 512)
size =  262144
min:  0.0
mean:  0.0
max:  0.0
dimension 2 """
#******** crpix ********
crpix = np.array([254.525, 257.675])
""" <class 'numpy.ndarray'>
>f4
shape =  (2,)
size =  2
min:  254.525
mean:  256.09998
max:  257.675
dimension 1 """
#******** fovpix ********
fovpix = 0.00028507045
""" <class 'numpy.float32'>
float32
shape =  ()
size =  1
min:  0.00028507045
mean:  0.00028507045
max:  0.00028507045
dimension 0 """
#******** frontinteg ********
frontinteg = 0
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  0
mean:  0.0
max:  0
dimension 0 """
#******** hlonlat ********
hlonlat = np.zeros(3)
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** imsize ********
imsize = np.array([512,512], dtype=int)
""" <class 'numpy.ndarray'>
>i4
shape =  (2,)
size =  2
min:  512
mean:  512.0
max:  512
dimension 1 """
#******** limbdark ********
limbdark = 0.58
""" <class 'numpy.float32'>
float32
shape =  ()
size =  1
min:  0.58
mean:  0.58
max:  0.58
dimension 0 """
#******** losnbp ********
losnbp = 64
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  64
mean:  64.0
max:  64
dimension 0 """
#******** losrange ********
losrange = np.array([-10, 10], dtype=float)
""" <class 'numpy.ndarray'>
>f4
shape =  (2,)
size =  2
min:  -10.0
mean:  0.0
max:  10.0
dimension 1 """
#******** maxsubdiv ********
maxsubdiv = 4
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  4
mean:  4.0
max:  4
dimension 0 """
#******** mmlat ********
mmlat = np.zeros(2)
""" <class 'numpy.ndarray'>
>f4
shape =  (2,)
size =  2
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** mmlon ********
mmlon = np.zeros(2)
""" <class 'numpy.ndarray'>
>f4
shape =  (2,)
size =  2
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** modelid ********
modelid = 54
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  54
mean:  54.0
max:  54
dimension 0 """
#******** modparam ********
modparam = np.arange(100000.0)
""" <class 'numpy.ndarray'>
>f4
shape =  (10,)
size =  10
min:  0.0
mean:  10000.332
max:  100000.0
dimension 1 """
#******** neang ********
neang = np.zeros(3)
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** neonly ********
neonly = 0
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  0
mean:  0.0
max:  0
dimension 0 """
#******** nepos ********
nepos = np.zeros(3)
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** nerotang ********
nerotang = np.zeros(3)
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** nerotaxis ********
nerotaxis = np.array([3,2,1], dtype=int)
""" <class 'numpy.ndarray'>
>i4
shape =  (3,)
size =  3
min:  1
mean:  2.0
max:  3
dimension 1 """
#******** nerotcntr ********
nerotcntr = np.zeros(3)
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** netot ********
netot = np.zeros(shape=(512, 512))
""" <class 'numpy.ndarray'>
>f4
shape =  (512, 512)
size =  262144
min:  0.0
mean:  0.0
max:  0.0
dimension 2 """
#******** netranslation ********
netranslation = np.zeros(3)
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** obsang ********(orientation of the observer)
#obsang = 
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  -0.0
mean:  0.00039369808
max:  0.00062409905
dimension 1 """
#******** obslonlat ********(position of the observer in Carrington
#coordinate. If set, then obspos is ignored. The optical
#axis always points toward the Sun center. Use obsang to
#change telescope orientation.)
#obslonlat = 
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  0.12662393
mean:  76.88172
max:  229.89352
dimension 1 """
#******** obslonlatflag ********
obslonlatflag = 1
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  1
mean:  1.0
max:  1
dimension 0 """
#******** obspos ********(position of the observer in the Sun basis)
#obspos = 
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  -214.0
mean:  -71.333336
max:  0.0
dimension 1 """
#******** occrad ********
occrad = 0.0
""" <class 'numpy.float32'>
float32
shape =  ()
size =  1
min:  0.0
mean:  0.0
max:  0.0
dimension 0 """
#******** pc ********
#pc = 
""" <class 'numpy.ndarray'>
>f4
shape =  (2, 2)
size =  4
min:  -8.687118e-14
mean:  0.5
max:  1.0
dimension 2 """
#******** pofinteg ********
pofinteg = 0
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  0
mean:  0.0
max:  0
dimension 0 """
#******** poiang ********
poiang = np.zeros(3)
""" <class 'numpy.ndarray'>
>f4
shape =  (3,)
size =  3
min:  0.0
mean:  0.0
max:  0.0
dimension 1 """
#******** projtypecode ********
projtypecode = 2
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  2
mean:  2.0
max:  2
dimension 0 """
#******** pv2_1 ********
pv2_1 = 0.0
""" <class 'numpy.float64'>
float64
shape =  ()
size =  1
min:  0.0
mean:  0.0
max:  0.0
dimension 0 """
#******** quiet ********
quiet = 1
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  1
mean:  1.0
max:  1
dimension 0 """
#******** rho ********
rho = np.zeros(shape=(512, 512))
""" <class 'numpy.ndarray'>
>f4
shape =  (512, 512)
size =  262144
min:  0.0
mean:  0.0
max:  0.0
dimension 2 """
#******** roi ********
roi = np.ones(shape=(512,512), dtype=int)
""" <class 'numpy.ndarray'>
>i4
shape =  (512, 512)
size =  262144
min:  1
mean:  1.0
max:  1
dimension 2 """
#******** rotmat ********
rotmat = np.zeros(shape=(3,3))
""" <class 'numpy.ndarray'>
>f4
shape =  (3, 3)
size =  9
min:  0.0
mean:  0.0
max:  0.0
dimension 2 """
#******** rrr ********
rrr = np.zeros(shape=(512, 512))
""" <class 'numpy.ndarray'>
>f4
shape =  (512, 512)
size =  262144
min:  0.0
mean:  0.0
max:  0.0
dimension 2 """
#******** uvinteg ********
uvinteg = 0
""" <class 'numpy.int32'>
int32
shape =  ()
size =  1
min:  0
mean:  0.0
max:  0
dimension 0
PRINTING OUTPUTS """
