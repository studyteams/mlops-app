from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # path("admin/", admin.site.urls),
    path("", include("mymlopsapp1.urls", namespace="mymlopsapp1")),
]

# 개발 환경에서 미디어 파일을 서빙하기 위한 설정 (운영 환경에서는 Nginx 등으로 처리)
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
