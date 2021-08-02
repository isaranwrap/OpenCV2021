import Augmentor

/Users/Praveens/Desktop/ishan/OpenCV2021/data/lfw
p = Augmentor.Pipeline("/Users/Praveens/Desktop/ishan/OpenCV2021/data/lfw/Aaron_Sorkin")
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.sample(10000)