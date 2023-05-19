import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('generated_qa.csv')
df =df.dropna()

questions = df['Question'].tolist()
answers = df['Answer'].tolist()
qa = zip(questions,answers)
# Display the first few rows of the DataFrame
with open("generated_qa.txt", "w") as file:
    # Write the content to the file
    
    for q1,a1 in qa:
        try:
            l =q1.split(sep="<sep>")
        except:
            print("Error",q1)
        st = [k.strip() for k in l]
        unique_list = list(set(st))
        for q2 in unique_list:
            if len(q2) >20:
                file.write('\n' +q2 +'\n'+a1)
        