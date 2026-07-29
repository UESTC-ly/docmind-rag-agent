from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.document import (
    DocumentChunkResponse,
    DocumentMetadataUpdate,
    DocumentResponse,
    UploadResponse,
)
from app.services import document_service
from app.utils.deps import get_current_user

router = APIRouter(prefix="/documents", tags=["文档"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档（异步解析）",
)
async def upload(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await document_service.create_document(db, current_user.id, file)
    return UploadResponse(document_id=doc.id, status=doc.status)


@router.get("/", response_model=list[DocumentResponse], summary="我的文档列表")
async def list_docs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await document_service.list_documents(db, current_user.id)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="查询文档解析状态",
)
async def get_doc(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await document_service.get_document(db, current_user.id, document_id)


@router.patch(
    "/{document_id}/metadata",
    response_model=DocumentResponse,
    summary="更新文档来源、版本和有效期",
)
async def update_metadata(
    document_id: int,
    metadata: DocumentMetadataUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await document_service.update_document_metadata(
        db,
        current_user.id,
        document_id,
        metadata,
    )


@router.get(
    "/{document_id}/chunks/{chunk_index}",
    response_model=DocumentChunkResponse,
    summary="跳转到引用对应的原文片段",
)
async def get_chunk(
    document_id: int,
    chunk_index: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await document_service.get_document_chunk(
        db,
        current_user.id,
        document_id,
        chunk_index,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除文档",
)
async def delete_doc(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await document_service.delete_document(db, current_user.id, document_id)
