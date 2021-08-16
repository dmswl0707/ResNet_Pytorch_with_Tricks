from training import *
#from k_fold import *

if __name__ == "__main__":

    training()
    #fold_train(Args, train_df)



    # 웨이트 저장하기

    PATH = './weights/'
    #torch.save(model.state_dict(), PATH +  'model_state_dict3.pt')