from DataLoader import testloader, categories
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from model import *
from training import *


PATH = '/home/eunji/project_dir/cifar+ResNet/custom_model.pt'

model = model.to(device)
model.cuda()


nb_classes = 10
confusion_matrix = np.zeros((nb_classes, nb_classes))
model.load_state_dict(torch.load(PATH))

model.eval()

with torch.no_grad():
    for i, (inputs, classes) in enumerate(testloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


plt.figure(figsize=(25,20))
print(confusion_matrix)

class_names = list(categories)
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()