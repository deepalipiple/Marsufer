from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
from markupsafe import escape
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import statistics as st
from flask_mysqldb import MySQL
import mysql.connector


app = Flask(__name__)
app.secret_key = 'my_secret_key'

# MySQL configuration
mysql_connection = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Deepali@123",
  database="flask_db"
)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Get form data
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # Check if username or email already exists
        cursor = mysql_connection.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username=%s OR email=%s", (username, email))
        user = cursor.fetchone()

        if user:
            error = "Username or email already exists"
            return render_template("register.html", error=error)
        else:
            # Insert new user into database
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
            mysql_connection.commit()
            session["username"] = username
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Get form data
        username = request.form["username"]
        password = request.form["password"]

        # Check if user exists and password is correct
        cursor = mysql_connection.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()

        if user:
            session["username"] = username
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password"
            return render_template("login.html", error=error)

    return render_template("login.html")

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # get form data
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']

        # save form data to database
        cur = mysql_connection.cursor()
        cur.execute("INSERT INTO contact_info (name, email,phone, message) VALUES (%s, %s, %s, %s)", (name, email,phone, message))
        mysql_connection.commit()
        cur.close()

        # display success message
        return render_template('thanks.html')
    else:
        return render_template('contact.html')
    

# profile route
@app.route('/profile')
def profile():
    if 'username' in session:
        # get the username and email from the session
        username = session['username']
        email = session['email']
        # render the profile template and pass the variables
        return render_template('profile.html', username=username, email=email)
    else:
        # redirect to the login page if the user is not logged in
        return redirect(url_for('login'))


# logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# home page route
@app.route('/')
def home():
    # if 'username' in session:
        # code to render the home page for logged-in users
       # return render_template('index.html', username=session['username'])
    # else:
        # code to redirect to the login page for non-logged-in users
        # return redirect(url_for('register'))
        return render_template("index.html")

@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    i=0

    GR_dict={0:(0,255,0),1:(0,0,255)}
    model = tf.keras.models.load_model('final_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output=[]
    cap = cv2.VideoCapture(0)
    while (i<=30):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img,1.05,5)

        for x,y,w,h in faces:

            face_img = img[y:y+h,x:x+w] 

            resized = cv2.resize(face_img,(224,224))
            reshaped=resized.reshape(1, 224,224,3)/255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)
            
            
            
            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
            cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27: 
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.median(output)
    return render_template("buttons.html",final_output=final_output1)


@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")


@app.route('/features')
def features():
    return render_template("features.html")



@app.route('/team')
def team():
    return render_template("team.html")


@app.route('/moviesAngry')
def moviesAngry():
    return render_template("moviesAngry.html")




if __name__ =='__main__':
    app.run(host="localhost", port=8000, debug=True)
