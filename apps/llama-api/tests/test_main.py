import pytest
from app import ray_init, dispatch_request_to_model, predict_cpu, predict_gpu
from fastapi.testclient import TestClient
from app.protocol.openai_api_protocol import CompletionRequest

def test_ray_init():
    # Test if the Ray initialization is done properly by checking the address and namespace 
    ray = ray_init()
    assert ray.address == "http://localhost:8000"
    assert ray.namespace == "kuberay"

@pytest.fixture(scope="module")
def test_client():
    from app import app
    client = TestClient(app)
    yield client  # testing happens here

def test_index(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Message": "This is Index"}

@pytest.mark.asyncio
async def test_predict():
    request = CompletionRequest(prompt="Test prompt", model="tiiuae/falcon-7b-instruct")
    
    # Test CPU prediction
    cpu_response = await predict_cpu(request)
    assert isinstance(cpu_response, dict)
    assert 'model' in cpu_response
    assert 'choices' in cpu_response
    assert 'usage' in cpu_response
    
    # Test GPU prediction
    gpu_response = await predict_gpu(request)
    assert isinstance(gpu_response, dict)
    assert 'model' in gpu_response
    assert 'choices' in gpu_response
    assert 'usage' in gpu_response