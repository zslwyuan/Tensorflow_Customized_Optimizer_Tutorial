
   Thanks for Prof. Yeung's detailed lectures and TAs' patient guide!
   Following are the guide to
   test my project. You can also access following links
   to watch some basic DEMOs videos of my project. There are detailed comments in the source code and the [report](https://github.com/zslwyuan/Tensorflow_Customized_Optimizer_Tutorial/blob/master/prj_report.pdf) will also help for understanding.
   
   
CNN DEMO:                         https://www.youtube.com/watch?v=yarDP2YRYok

CAE DEMO(Learning Rate = 0.1):    https://www.youtube.com/watch?v=I5Z8MNnHnKc

CAE DEMO(Learning Rate = 0.001):  https://www.youtube.com/watch?v=6aa1Vx_Z8Hg


GUIDE:

1. Data Directory:
   please extract the compressed file 
   and copy the datasets (e.g.data_autoencoder_eval.npz) 
   into the sub-directory "data", which is in the same
   directory of the Python files 
   (
        The python files find the dataset in such way 
        np.load("./data/xxxxxx.npz")
   )
   
2. task1_plot.py : CNN Training and Evaluation

    a) run the python file and let it plot figures for classified samples
        type: python task1_plot.py
        
    a) run the python file and disbale it to plot figures for classified samples
        type: python task1_plot.py --nofigure
        
3. task2_plot.py : CAE Training and Evaluation

    a) run the python file and let it plot figures for noise-inserted samples,
        reconstructed samples and features maps (encoded layer)
        type: python task2_plot.py
        
    a) run the python file and disbale it to plot figures for those figures
        type: python task2_plot.py --nofigure
        
4. task1_cv.py : Cross-Validation for CNN Training

        type: python task1_cv.py
        
5. task2_cv.py : Cross-Validation for CAE Training

        type: python task2_cv.py
        
