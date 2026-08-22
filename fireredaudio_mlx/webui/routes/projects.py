"""Project management API routes."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..workspace import workspace_store, Project

router = APIRouter(prefix="/projects", tags=["Projects"])


class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.get("", response_model=List[Project])
def list_projects():
    return workspace_store.list_projects()


@router.post("", response_model=Project)
def create_project(req: CreateProjectRequest):
    return workspace_store.create_project(name=req.name, description=req.description)


@router.get("/{project_id}", response_model=Project)
def get_project(project_id: str):
    p = workspace_store.get_project(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="找不到指定的项目")
    return p


@router.patch("/{project_id}", response_model=Project)
def update_project(project_id: str, req: UpdateProjectRequest):
    p = workspace_store.update_project(project_id, name=req.name, description=req.description)
    if not p:
        raise HTTPException(status_code=404, detail="找不到指定的项目")
    return p


@router.delete("/{project_id}")
def delete_project(project_id: str):
    ok = workspace_store.delete_project(project_id)
    if not ok:
        raise HTTPException(status_code=404, detail="找不到指定的项目")
    return {"message": "项目已删除", "project_id": project_id}
