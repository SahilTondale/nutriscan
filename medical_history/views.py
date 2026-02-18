from django.shortcuts import render, redirect
from .models import MedicalHistory
from django.contrib.auth.decorators import login_required
from .forms import MedicalHistoryForm

# Create your views here.
@login_required
def medical_history_view(request):
    try:
        # Check if medical history exists
        medical_history = request.user.medicalhistory
        return redirect('upload')
    except MedicalHistory.DoesNotExist:
        pass
    
    if request.method == 'POST':
        form = MedicalHistoryForm(request.POST)
        if form.is_valid():
            medical_history = form.save(commit=False)
            medical_history.user = request.user
            medical_history.save()
            return redirect('upload')
    else:
        form = MedicalHistoryForm()
    
    return render(request, 'medical_history/form.html', {'form': form})

@login_required
def check_medical_history(request):
    if hasattr(request.user, 'medicalhistory'):
        return redirect('upload')
    return redirect('medical_history')

@login_required
def edit_medical_info(request):
    medical_history, created = MedicalHistory.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = MedicalHistoryForm(request.POST, instance=medical_history)
        if form.is_valid():
            form.save()
            return redirect('upload')
    else:
        form = MedicalHistoryForm(instance=medical_history)

    return render(request, 'medical_history/edit.html', {'form': form})