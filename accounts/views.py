from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.models import User, Group
from django.contrib.auth import authenticate
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages

from .forms import CreateAccountForm


@login_required
@permission_required("auth.add_user", raise_exception=True)
def account_list(request):
    users = User.objects.all().order_by("username")
    return render(request, "accounts/account_list.html", {"users": users})


@login_required
@permission_required("auth.add_user", raise_exception=True)
def create_account(request):
    if request.method == "POST":
        form = CreateAccountForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            user = User.objects.create_user(
                username=data["username"],
                email=data["email"],
                password=data["password1"],
                is_staff=True,  # required to log into any @login_required admin page
            )
            if data["role"] in ("Admin", "both"):
                user.groups.add(Group.objects.get(name="Admin"))
            if data["role"] in ("Approver", "both"):
                user.groups.add(Group.objects.get(name="Approver"))
            messages.success(request, f"Account '{user.username}' created.")
            return redirect("accounts:account_list")
    else:
        form = CreateAccountForm()

    return render(request, "accounts/create_account.html", {"form": form})


@login_required
@permission_required("auth.change_user", raise_exception=True)
def toggle_active(request, pk):
    target_user = get_object_or_404(User, pk=pk)

    if target_user == request.user:
        messages.error(request, "You can't deactivate your own account.")
        return redirect("accounts:account_list")

    if request.method == "POST":
        target_user.is_active = not target_user.is_active
        target_user.save()
        status = "activated" if target_user.is_active else "deactivated"
        messages.success(request, f"Account '{target_user.username}' {status}.")

    return redirect("accounts:account_list")


@login_required
@permission_required("auth.delete_user", raise_exception=True)
def delete_account(request, pk):
    target_user = get_object_or_404(User, pk=pk)
    error = None

    if target_user == request.user:
        messages.error(request, "You can't delete your own account.")
        return redirect("accounts:account_list")

    if request.method == "POST":
        password = request.POST.get("password", "")
        confirming_user = authenticate(
            request, username=request.user.username, password=password
        )
        if confirming_user is None:
            error = "Incorrect password. Deletion cancelled."
        else:
            username = target_user.username
            target_user.delete()
            messages.success(request, f"Account '{username}' deleted.")
            return redirect("accounts:account_list")

    return render(
        request,
        "accounts/delete_account_confirm.html",
        {"target_user": target_user, "error": error},
    )
