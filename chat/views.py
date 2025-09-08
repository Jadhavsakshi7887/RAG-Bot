from django.shortcuts import render
from django.conf import settings
from gemini import initialize_vectorstore, rag_query
import os

def chat_view(request):
    answer = None
    uploaded_file_name = None

    if request.method == "POST":
        question = request.POST.get("question")
        uploaded_file = request.FILES.get("document")

        if uploaded_file:
            temp_dir = os.path.join(settings.BASE_DIR, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb+") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            request.session['uploaded_file_path'] = file_path
            request.session['uploaded_file_name'] = uploaded_file.name

            initialize_vectorstore(file_path)

        uploaded_file_name = request.session.get('uploaded_file_name')
        file_path = request.session.get('uploaded_file_path')

        if question:
            if file_path:
                initialize_vectorstore(file_path)
                answer = rag_query(question)
            else:
                answer = "Please upload a file first."

    return render(request, "chat/index.html", {
        "answer": answer,
        "uploaded_file_name": uploaded_file_name
    })