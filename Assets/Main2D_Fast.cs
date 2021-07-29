using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;


public class Main2D_Fast : MonoBehaviour
{
    static int particleCount;
    static int particleCountPow2;
    static int wallsCount;

    static float h = 0.3f;
    static float rho0 = 1f;
    static float particleRadius = 0.05f;
    static float K = particleRadius * 600f;
    public float viscosity = 0.018f * 25;
    public float G = 9.81f;

    static int stepsPerFrame = 1;

    static int threadCount = 256;
    static int partitionBucketCount = 128 * 128 * 4;

    static float Poly6_constant = (315.0f / (64.0f * 3.1415f * Mathf.Pow(h,9)));
    static float Spiky_constant = (-45f / (3.1415f * Mathf.Pow(h, 6)));


    Particle[] particles;
    Wall[] walls;

    ComputeBuffer buffer;
    ComputeBuffer particleIndexBuffer;
    ComputeBuffer cellIndexBuffer;
    ComputeBuffer cellOffsetBuffer;
    ComputeBuffer cellIndexBufferSorted;
    ComputeBuffer particleIndexBufferSorted;
    ComputeBuffer wallsBuffer;



    public ComputeShader BitonicSortShader;
    int BUFFER_SIZE;
    const uint BITONIC_BLOCK_SIZE = 256;
    const uint TRANSPOSE_BLOCK_SIZE = 16;

    const int KERNEL_ID_BITONICSORT = 0;
    const int KERNEL_ID_TRANSPOSE_MATRIX = 1;

    ComputeBuffer particleIndexTempBuffer;
    ComputeBuffer cellIndexTempBuffer;


    public ComputeShader shader;
    public ComputeShader renderShader;


    int kernel1;
    int kernel2;
    int kernel3;
    int kernel_grid1;
    int kernel_grid2;
    int kernel_grid3;
    int kernel_render;

    List<Particle> fluidParticlesList = new List<Particle>();
    List<Wall> wallsList = new List<Wall>();

    public GameObject wallContainer;
    public GameObject lavaContainer;

    private RenderTexture _target;
    private Camera _camera;


    public struct Particle
    {
        public float mass;
        public float density;
        public float pressure;
        public int alive;

        public Vector2 position;
        public Vector2 velocity;
        public Vector2 pressureForce;
        public Vector2 viscosityForce;
    };

    public struct Wall
    {
        public Vector2 origin;
        public Vector2 size;
    };

    private void Awake()
    {
        _camera = Camera.main;
    }

    void Start()
    {
        InitializeParticles();
        InitializeShader();
    }

    void Update()
    {
        RunShader();
    }

    void InitializeParticles()
    {
        // Prova 64 x 64. Lo spacing è h * 8/100, quindi se voglio x = 64 * (h * 8/100 ) = 1.536
        for (int i = 0; i < lavaContainer.transform.childCount; i++)
        {
            Transform child = lavaContainer.transform.GetChild(i);
            CreateFluidBlock(child.position, child.localScale);
        }

        wallsList = new List<Wall>();

        for (int i = 0; i < wallContainer.transform.childCount; i++)
        {
            Transform child = wallContainer.transform.GetChild(i);

            Wall wall = new Wall {
                origin = new Vector2(child.position.x, child.position.y),
                size = new Vector2(child.localScale.x, child.localScale.y) + new Vector2(1f, 1f) * 0.2f
            };
            wallsList.Add(wall);
        }
        walls = wallsList.ToArray();
        wallsCount = walls.Length;

        // Traasformo queste liste in un array statico che le comprende tutte
        List<Particle> allParticles = new List<Particle>();
        allParticles.AddRange(fluidParticlesList);

        // Il numero di particelle allocate sulla GPU deve essere una potenza di 2
        particleCount = fluidParticlesList.Count;
        particleCountPow2 = ToNextNearest(particleCount);
        particleCountPow2 = Mathf.Max(particleCountPow2, threadCount);

        Debug.Log(particleCount);

        // Aggiungo particelle finte alla lista
        for (int i = 0; i < particleCountPow2 - particleCount; i++) allParticles.Add(new Particle { position = new Vector2(0, -100), alive = 0});
    
        // Trasformo la lista in un array
        particles = allParticles.ToArray();
    }

    void InitializeShader()
    {
        // Creo un buffer per passare le info sulle particelle
        buffer = new ComputeBuffer(particleCountPow2, (1 + 1 + 1 + 1 + 2 + 2 + 2 + 2) * 4);
        buffer.SetData(particles);

        // Creo i buffer per la gestione delle celle
        particleIndexBuffer = new ComputeBuffer(particleCountPow2, 4);
        cellIndexBuffer = new ComputeBuffer(particleCountPow2, 4);
        cellOffsetBuffer = new ComputeBuffer(partitionBucketCount, 4); 
        cellIndexBufferSorted = new ComputeBuffer(particleCountPow2, 4);
        particleIndexBufferSorted = new ComputeBuffer(particleCountPow2, 4);
        wallsBuffer = new ComputeBuffer(wallsCount, 4 * 4);

        // Identifico i kernel dello shader
        kernel1 = shader.FindKernel("ComputeDensityAndPressure");
        kernel2 = shader.FindKernel("ComputeForces");
        kernel3 = shader.FindKernel("UpdateParticles");

        kernel_grid1 = shader.FindKernel("GetCellIndex");
        kernel_grid2 = shader.FindKernel("GetOffsetList");
        kernel_grid3 = shader.FindKernel("ResetOffsetList");

        // Passo il buffer ai kernel dei compute shader
        shader.SetBuffer(kernel1, "particleBuffer", buffer);
        shader.SetBuffer(kernel2, "particleBuffer", buffer);
        shader.SetBuffer(kernel3, "particleBuffer", buffer);

        shader.SetBuffer(kernel1, "particleIndexBuffer", particleIndexBuffer);
        shader.SetBuffer(kernel1, "cellOffsetBuffer", cellOffsetBuffer);
        shader.SetBuffer(kernel1, "cellIndexBuffer", cellIndexBuffer);
        shader.SetBuffer(kernel2, "particleIndexBuffer", particleIndexBuffer);
        shader.SetBuffer(kernel2, "cellOffsetBuffer", cellOffsetBuffer);
        shader.SetBuffer(kernel2, "cellIndexBuffer", cellIndexBuffer);
        shader.SetBuffer(kernel3, "particleIndexBuffer", particleIndexBuffer);
        shader.SetBuffer(kernel3, "cellOffsetBuffer", cellOffsetBuffer);
        shader.SetBuffer(kernel3, "cellIndexBuffer", cellIndexBuffer);

        shader.SetBuffer(kernel_grid1, "particleIndexBuffer", particleIndexBuffer);
        shader.SetBuffer(kernel_grid1, "cellIndexBuffer", cellIndexBuffer);
        shader.SetBuffer(kernel_grid1, "particleBuffer", buffer);
        shader.SetBuffer(kernel_grid2, "particleIndexBuffer", particleIndexBuffer);
        shader.SetBuffer(kernel_grid2, "cellIndexBuffer", cellIndexBuffer);
        shader.SetBuffer(kernel_grid2, "cellOffsetBuffer", cellOffsetBuffer);
        shader.SetBuffer(kernel_grid3, "cellOffsetBuffer", cellOffsetBuffer);


        shader.SetBuffer(kernel3, "wallsBuffer", wallsBuffer);
        shader.SetInt("wallsCount", wallsCount);
        wallsBuffer.SetData(walls);

        // Setto i parametri globali
        shader.SetInt("partitionBucketCount", partitionBucketCount);
        shader.SetInt("particleCount", particleCount);
        shader.SetFloat("h", h);
        shader.SetFloat("h2", h * h);
        shader.SetFloat("Poly6_constant", Poly6_constant);
        shader.SetFloat("Spiky_constant", Spiky_constant);

        shader.SetFloat("rho0", rho0);
        shader.SetFloat("K", K);

        shader.SetFloat("dt", 0.016f / (float)stepsPerFrame);
        shader.SetVector("G", new Vector4(0,-G));
        shader.SetFloat("viscosity", viscosity);
       
        // Creo gli array per memorizzare questi dati (inutile se faccio il sort su GPU, ma per ora..)
        uint[] particleIndexArray = new uint[particleCountPow2];
        for (int i = 0; i < particleCountPow2; i++) particleIndexArray[i] = (uint)i;
        particleIndexBuffer.SetData(particleIndexArray);


        uint[] cellIndexArray = new uint[particleCountPow2];
        for (int i = 0; i < particleCountPow2; i++)
        {
            if (i < particleCount)
                cellIndexArray[i] = 0;
            else
                cellIndexArray[i] = 0xFFFFFFFF;
        }
        cellIndexBuffer.SetData(cellIndexArray);


        kernel_render = renderShader.FindKernel("RenderFluid");
        renderShader.SetBuffer(kernel_render, "particleBuffer", buffer);
        renderShader.SetBuffer(kernel_render, "particleIndexBuffer", particleIndexBuffer);
        renderShader.SetBuffer(kernel_render, "cellOffsetBuffer", cellOffsetBuffer);
        renderShader.SetBuffer(kernel_render, "cellIndexBuffer", cellIndexBuffer);

        renderShader.SetInt("particleCount", particleCount);
        renderShader.SetInt("partitionBucketCount", partitionBucketCount);
        renderShader.SetFloat("h", h);


        renderShader.SetMatrix("_CameraToWorld", _camera.cameraToWorldMatrix);
        renderShader.SetMatrix("_CameraInverseProjection", _camera.projectionMatrix.inverse);


        BUFFER_SIZE = particleCountPow2;
        particleIndexTempBuffer = new ComputeBuffer(BUFFER_SIZE, Marshal.SizeOf(typeof(uint)));
        cellIndexTempBuffer = new ComputeBuffer(BUFFER_SIZE, Marshal.SizeOf(typeof(uint)));

    }

    void RunShader()
    {
        shader.SetVector("G", new Vector4(0, -G));
        shader.SetFloat("viscosity", viscosity);

        // Eseguo uno step di evoluzione
        for (int n = 0; n < stepsPerFrame; ++n)
        {
            // Valuto l'index con hash delle celle
            shader.Dispatch(kernel_grid1, particleCountPow2 / threadCount, 1, 1);

            GPUSort();

            // Resetto la lista degli offset
            shader.Dispatch(kernel_grid3, partitionBucketCount / threadCount, 1, 1);

            // Valuto la lista di offset
            shader.Dispatch(kernel_grid2, particleCountPow2 / threadCount, 1, 1);
             

            // Prima calcolo densità e pressione     
            shader.Dispatch(kernel1, particleCountPow2 / threadCount, 1, 1);

            // Poi valuto le forze
            shader.Dispatch(kernel2, particleCountPow2 / threadCount, 1, 1);
            
            // Poi modifico la posizione delle particelle
            shader.Dispatch(kernel3, particleCountPow2 / threadCount, 1, 1);
                      
        }
    }


    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Render(source,destination);
    }

    private void Render(RenderTexture source, RenderTexture destination)
    {

        renderShader.SetMatrix("_CameraToWorld", _camera.cameraToWorldMatrix);
        renderShader.SetMatrix("_CameraInverseProjection", _camera.projectionMatrix.inverse);

        // Make sure we have a current render target
        InitRenderTexture();

        // Set the target and dispatch the compute shader
        renderShader.SetTexture(kernel_render, "Result", _target);
        renderShader.SetTexture(kernel_render, "sourceTexture", source);

        int threadGroupsX = Mathf.CeilToInt(Screen.width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(Screen.height / 8.0f);

        renderShader.Dispatch(kernel_render, threadGroupsX, threadGroupsY, 1);

        // Blit the result texture to the screen
        Graphics.Blit(_target, destination);
    }

    private void InitRenderTexture()
    {
        if (_target == null || _target.width != Screen.width || _target.height != Screen.height)
        {
            // Release render texture if we already have one
            if (_target != null)
                _target.Release();

            // Get a render target for Ray Tracing
            _target = new RenderTexture(Screen.width, Screen.height, 0,
                RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            _target.enableRandomWrite = true;
            _target.Create();
        }
    }



    public void CreateFluidBlock(Vector2 origin, Vector2 size)
    {
        float dx = particleRadius;
        int nx = (int)Mathf.Ceil(size.x / dx);
        int ny = (int)Mathf.Ceil(size.y / dx);

        for (int x = 0; x < nx; x++)
        {
            for (int y = 0; y < ny; y++)
            {

                Vector2 pos = origin + new Vector2(x, y) * dx - size/2.0f + 0*(Random.value-0.5f)*0.01f * Vector2.one;
                float mass = rho0 * (dx/2.0f) * (dx/2.0f);
       
                Particle p = new Particle
                {
                    mass = mass,
                    density = rho0,
                    pressure = 0,
                    alive = 1,
                    position = pos,
                    velocity = Vector2.zero,
                    pressureForce = Vector2.zero,
                    viscosityForce = Vector2.zero
                };

                fluidParticlesList.Add(p);               
            }
        }
    }

    public int ToNextNearest(int x)
    {
        if (x < 0) { return 0; }
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    }







    void GPUSort()
    {
        ComputeShader shader = BitonicSortShader;
        // Determine parameters.
        uint NUM_ELEMENTS = (uint)BUFFER_SIZE;
        uint MATRIX_WIDTH = BITONIC_BLOCK_SIZE;
        uint MATRIX_HEIGHT = (uint)NUM_ELEMENTS / BITONIC_BLOCK_SIZE;

        // Sort the data
        // First sort the rows for the levels <= to the block size
        for (uint level = 2; level <= BITONIC_BLOCK_SIZE; level <<= 1)
        {
            SetGPUSortConstants(shader, level, level, MATRIX_HEIGHT, MATRIX_WIDTH);

            // Sort the row data
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data1", cellIndexBuffer);
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data2", particleIndexBuffer);

            shader.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);
        }
        
        // Then sort the rows and columns for the levels > than the block size
        // Transpose. Sort the Columns. Transpose. Sort the Rows.
        for (uint level = (BITONIC_BLOCK_SIZE << 1); level <= NUM_ELEMENTS; level <<= 1)
        {
            // Transpose the data from buffer 1 into buffer 2
            SetGPUSortConstants(shader, (level / BITONIC_BLOCK_SIZE), (level & ~NUM_ELEMENTS) / BITONIC_BLOCK_SIZE, MATRIX_WIDTH, MATRIX_HEIGHT);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Input1", cellIndexBuffer);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Input2", particleIndexBuffer);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Data1", cellIndexTempBuffer);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Data2", particleIndexTempBuffer);
            shader.Dispatch(KERNEL_ID_TRANSPOSE_MATRIX, (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE), (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE), 1);

            // Sort the transposed column data
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data1", cellIndexTempBuffer);
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data2", particleIndexTempBuffer);
            shader.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);

            // Transpose the data from buffer 2 back into buffer 1
            SetGPUSortConstants(shader, BITONIC_BLOCK_SIZE, level, MATRIX_HEIGHT, MATRIX_WIDTH);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Input1", cellIndexTempBuffer);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Input2", particleIndexTempBuffer);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Data1", cellIndexBuffer);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Data2", particleIndexBuffer);
            shader.Dispatch(KERNEL_ID_TRANSPOSE_MATRIX, (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE), (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE), 1);

            // Sort the row data
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data1", cellIndexBuffer);
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data2", particleIndexBuffer);
            shader.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);
        }
        
    }

    void SetGPUSortConstants(ComputeShader cs, uint level, uint levelMask, uint width, uint height)
    {
        cs.SetInt("_Level", (int)level);
        cs.SetInt("_LevelMask", (int)levelMask);
        cs.SetInt("_Width", (int)width);
        cs.SetInt("_Height", (int)height);
    }


}
