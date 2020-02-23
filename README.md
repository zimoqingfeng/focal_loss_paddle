# focal_loss_paddle
A Paddle Implementation of Focal Loss.

####Input: 
    logit: Softmax logit from deep network
    label: Groundtruth
    class_dim: Number of categories
    gamma & alpha: The settings of focal loss, and the alpha should be list
    smooth: the settings of label smoothing
    
####Output: 
    loss: the loss of focal loss implementation on Paddle-1.6.1