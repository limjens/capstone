from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = "accounts"

urlpatterns = [
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="accounts/login.html"),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("manage/", views.account_list, name="account_list"),
    path("manage/create/", views.create_account, name="create_account"),
    path("manage/<int:pk>/toggle/", views.toggle_active, name="toggle_active"),
    path("manage/<int:pk>/delete/", views.delete_account, name="delete_account"),
]
