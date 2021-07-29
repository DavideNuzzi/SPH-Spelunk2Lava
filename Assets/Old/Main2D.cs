using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Main2D : MonoBehaviour
{
    static int fluidParticleCount;
    static int wallParticleCount;
    static int particleCount;
    static int particleCountPow2;

    static int threadCount = 128;
    static float h = 0.3f;
    static float rho0 = 1.0f;

    static float Poly6_constant = (315.0f / (64.0f * 3.1415f * Mathf.Pow(h,9)));
    static float Spiky_constant = (-45f / (3.1415f * Mathf.Pow(h, 6)));

    static float viscosity = 0.018f*2;
    static int stepsPerFrame = 1;

    Particle[] particles;
    Particle[] output;

    ComputeBuffer buffer;
    public ComputeShader shader;
    int kernel1;
    int kernel2;
    int kernel3;


    public GameObject testParticleObject;
    public GameObject testWallObject;

    List<Particle> fluidParticlesList = new List<Particle>();
    List<Particle> wallParticlesList = new List<Particle>();
    GameObject[] particleObjects;


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
        CreateFluidBlock(0.25f*new Vector2(3, 3), 0.5f*new Vector2(3, 3));
        //  CreateFluidBlock(0.25f * new Vector2(3, 3) + new Vector2(0,10f), 0.1f * new Vector2(3, 3));
        CreateWall(new Vector2(2, 0.5f), 0.5f*new Vector2(1f,1f)); // Pavimento

        //  CreateFluidBlock(new Vector3(0, 2, 0), new Vector3(1, 1, 1) * 0.3f);

        /*
        CreateWall(new Vector3(0, 0, 0), new Vector3(2, 0.1f, 2)); // Pavimento
        CreateWall(new Vector3(-1, 0.5f, 0), new Vector3(0.1f, 1, 2)); // Muro SX
        CreateWall(new Vector3(1, 0.5f, 0), new Vector3(0.1f, 1, 2)); // Muro DX
        CreateWall(new Vector3(0, 0.5f, -1), new Vector3(2, 1, 0.1f)); // Muro Davanti
        CreateWall(new Vector3(0, 0.5f, 1), new Vector3(2, 1, 0.1f)); // Muro Dietro
        */
        //   CreateWall(new Vector3(0, 1, 0), new Vector3(1, 1, 1));

        // Traasformo queste liste in un array statico che le comprende tutte
        List<Particle> allParticles = new List<Particle>();
        allParticles.AddRange(fluidParticlesList);
        allParticles.AddRange(wallParticlesList);

        // Il numero di particelle allocate sulla GPU deve essere una potenza di 2
        fluidParticleCount = fluidParticlesList.Count;
        wallParticleCount = wallParticlesList.Count;
        particleCount = fluidParticleCount + wallParticleCount;
        particleCountPow2 = ToNextNearest(particleCount);
        Debug.Log(particleCount);
        Debug.Log(particleCountPow2);

        // Aggiungo particelle finte alla lista
        for (int i = 0; i < particleCountPow2 - particleCount; i++) allParticles.Add(new Particle { position = new Vector2(0, -100), isSolid = 1 });
    
        // Trasformo la lista in un array
        particles = allParticles.ToArray();

        // Genero i modelli per le particelle di entrambi i tipi
        particleObjects = new GameObject[particleCount];

        for (int i = 0; i < particleCount; i++)
        {
            if (particles[i].isSolid == 0)
            {
                particleObjects[i] = GameObject.Instantiate(testParticleObject, particles[i].position, Quaternion.identity);
            }
            else
            {
                particleObjects[i] = GameObject.Instantiate(testWallObject, particles[i].position, Quaternion.identity);
            }
        }

        output = new Particle[particleCount];
    }

    void InitializeShader()
    {
        // Creo un buffer per passare le info sulle particelle
        buffer = new ComputeBuffer(particleCountPow2, (1 + 1 + 1 + 1 + 2 + 2 + 2 + 2) * 4);
        buffer.SetData(particles);

        // Passo il buffer al kernel del compute shader
        kernel1 = shader.FindKernel("ComputeDensityAndPressure");
        kernel2 = shader.FindKernel("ComputeForces");
        kernel3 = shader.FindKernel("UpdateParticles");

        // Setto i parametri globali
        shader.SetBuffer(kernel1, "particleBuffer", buffer);
        shader.SetBuffer(kernel2, "particleBuffer", buffer);
        shader.SetBuffer(kernel3, "particleBuffer", buffer);

        shader.SetInt("particleCount", particleCount);
        shader.SetFloat("h", h);
        shader.SetFloat("h2", h * h);
        shader.SetFloat("h3", h * h * h);
        shader.SetFloat("Poly6_constant", Poly6_constant);
        shader.SetFloat("Spiky_constant", Spiky_constant);

        shader.SetFloat("rho0", rho0);
        shader.SetFloat("K", 30f);

        shader.SetFloat("dt", 0.016f / (float)stepsPerFrame);
        shader.SetVector("G", new Vector4(0,-9.81f));
        shader.SetFloat("viscosity", viscosity);
        
    }

    void RunShader()
    {
        // Eseguo uno step di evoluzione
        //    buffer.SetData(particles);
        for (int n = 0; n < stepsPerFrame; n++)
        {
            // Prima calcolo densità e pressione
            shader.Dispatch(kernel1, particleCountPow2 / threadCount, 1, 1);

            // Poi valuto le forze
            shader.Dispatch(kernel2, particleCountPow2 / threadCount, 1, 1);

            // Poi modifico la posizione delle particelle
            shader.Dispatch(kernel3, particleCountPow2 / threadCount, 1, 1);
        }
        // Estraggo le informazioni sulle particelle
        buffer.GetData(particles);

        // Mostro a schermo la nuova posizione delle particelle (ATTENZIONE! qui posso eliminare quelle dei muri)
        for (int i = 0; i < particleCount; i++)
        {
            particleObjects[i].transform.position = particles[i].position;
        }

        
        // Per il debug, ordino le forze di pressione e di attrito
        float[] pressureMagnitudes = new float[particleCount];
        float[] viscosityMagnitudes = new float[particleCount];
        float[] pressureValues = new float[particleCount];
        float[] densityValues = new float[particleCount];
        float[] velocityMagnitudes = new float[particleCount];

        for (int i = 0; i < particleCount; i++)
        {
            pressureMagnitudes[i] = particles[i].pressureForce.magnitude;
            viscosityMagnitudes[i] = particles[i].viscosityForce.magnitude;
            velocityMagnitudes[i] = particles[i].velocity.magnitude;
            pressureValues[i] = particles[i].pressure;
            densityValues[i] = particles[i].density;

          float p = pressureValues[i] / 50f;
        //    float p = velocityMagnitudes[i] / 3f;
            particleObjects[i].GetComponent<MeshRenderer>().material.color = new Color(p, 0, 1 - p);
            if (particles[i].isSolid == 1) particleObjects[i].GetComponent<MeshRenderer>().material.color = new Color(1-p,  p,0);

        }

        // Coloro le particelle in base al valore della pressione


       // Debug.Log("Forza pressione massima: " + Mathf.Max(pressureMagnitudes));
       // Debug.Log("Forza viscosità massima: " + Mathf.Max(viscosityMagnitudes));
        

    }


    public void CreateWall(Vector2 origin, Vector2 size)
    {
        CreateParticleBlock(origin, size, 1, h / 5f * 0.5f);
    }


    public void CreateFluidBlock(Vector2 origin, Vector2 size)
    {
        CreateParticleBlock(origin, size, 0, h/5f * 0.5f * 0.8f);
    }


    public void CreateParticleBlock (Vector2 origin, Vector2 size, int isSolid, float dx)
    {
        int nx = (int)Mathf.Ceil(size.x / dx);
        int ny = (int)Mathf.Ceil(size.y / dx);

        for (int x = 0; x < nx; x++)
        {
            for (int y = 0; y < ny; y++)
            {

                Vector2 pos = origin + new Vector2(x, y) * dx - size/2.0f + 0*(Random.value-0.5f)*0.01f * Vector2.one;

                float mass = 0.000125f*10f * 0.25f * 0.8f * 0.8f;
                if (isSolid == 1) mass = 0.0005f;

                float density = rho0;
                if (isSolid == 1) density = 10;

                Particle p = new Particle
                {
                    mass = mass,
                    density = density,
                    pressure = 0,
                    isSolid = isSolid,
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

    public struct Particle
    {
        public float mass;
        public float density;
        public float pressure;
        public int isSolid;

        public Vector2 position;
        public Vector2 velocity;
        public Vector2 pressureForce;
        public Vector2 viscosityForce;
    };

}
