import glob
import os
import json 

'''names_of_dir =  os.listdir(os.path.join(os.getcwd(),"data/TRAIN")) 
names_of_dir.remove(".DS_Store")

names_of_dir =  os.listdir(os.path.join(os.getcwd(),"data/TRAIN")) 

'''

'''import glob
import os

file_list = glob.glob(os.path.join(os.getcwd(), "FolderName", "*.txt"))

corpus = []

for file_path in file_list:
    with open(file_path) as f_input:
        corpus.append(f_input.read())'''

def cleanfiles(textfiles):
    textfiles = [sub.strip().split(" ")[2:] for sub in textfiles]
    cleanTextfileList = []
    for textfile in textfiles: 
        cleanTextfileList.append(" ".join(textfile).strip())
    return cleanTextfileList


       

def extractTextFromFileList(file_list_text):
    textfilesList = []
    for promptfile in file_list_text:
        with open(promptfile) as my_file:
            content = my_file.read()
            #print(content)
            if content:
                textfilesList.append(content)
    
    textfilesListClean = cleanfiles(textfilesList)

    return textfilesListClean

def JsonMaker(file_wav_list,textfilesListClean):
    listOfDict = []
    for ii in range(len(file_wav_list)):
        dictOfJSON = {}
        #print(ii)
        dictOfJSON["Text:"] = "".join(textfilesListClean[ii])
        dictOfJSON["Ljud:"] = file_wav_list[ii]
        listOfDict.append(dictOfJSON)
#print(listOfDict)

    json_object = json.dumps(listOfDict, indent = 4) 

    return json_object

##### 

# To do: Skapa en se till så att det inte är en lista med WAW
# Sätta ut så det blir min egna path - se line 113 för försök till detta! 
# Skriva ut i fil! 


#####

## Fredags-tankar
# bara göra en som har samtliga prompts men bara en röst? Isf strunta i alla DR's och bara köra DR7. Bara Male voice kansk? 


####


#print(listOfAllDir)
## Ska bli i detta format ./data/TRAIN/DR1/M*/.WAV'
#filepathMDirectories = (r'./data/TRAIN/DR7/M*')





def main():
    fixedPathM = (r'./data/TRAIN/DR7/MWRP0')
    fixedPathWaw = os.path.join(fixedPathM,"*.WAV")
    fixedPathText = os.path.join(fixedPathM,"*.TXT")
    file_list = []

    #paths = glob.glob(os.path.join(currentDir,(r'.\/data\/TRAIN\/DR7\/M....\/.*.WAW')))
    file_list_waw = glob.glob(fixedPathWaw)
    file_list_text = glob.glob(fixedPathText)
    #print(file_list_waw)
    #print(file_list_text)
    #print(paths)

    listOfTextClean= extractTextFromFileList(file_list_text)
    #print(listOfTextClean)

    listOfJson = JsonMaker(file_list_waw,listOfTextClean)
    print(listOfJson)

main()



