# Extract quiz data from the response
import streamlit as st
import pandas as pd
with st.form("user_test"): 
        user_answers =[]
        df = pd.read_csv("out.csv")
        for index, row in df.iterrows():
                        st.subheader(f"{row['MCQ']}")
                        choices = row['Choices'].split(' | ')
                        selected_option = st.selectbox("Choose your answer:", choices)
                        user_answers.append(selected_option)    
        print(user_answers)
        user_answers = [string[0] for string in user_answers]
        #Submit button
        result_button = st.form_submit_button("Result")
        if result_button: #and user_answers is not None:
                    # Calculate score
                    print("I am here @@@@@@@@@@@@@@@@@@@@@@@@@")
                    print(df.head(2))
                    correct_answers = df['Correct'].tolist()
                    print(correct_answers)
                    print(user_answers)
                    #print(correct_answers)
                    score = sum(user_answer == correct_answer for user_answer, correct_answer in zip(user_answers, correct_answers))        
                    print("I am here also 2 @@@@@@@@@@@@@@@@@@@@@@@@@")
                    # Display score
                    print(f"Your Score: {score}/{len(df)}")
                    st.write(f"Your Score: {score}/{len(df)}")