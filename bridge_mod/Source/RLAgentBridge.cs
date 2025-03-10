using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using ICities;
using UnityEngine;
using ColossalFramework;
using ColossalFramework.UI;
using ColossalFramework.Plugins;

namespace CS2_RLAgent_Bridge
{
    public class RLAgentBridge : IUserMod, ILoadingExtension, IThreadingExtension
    {
        // API server configuration
        private const int ApiPort = 5000;
        private const string ApiHost = "localhost";
        private HttpListener listener;
        private Thread listenerThread;
        private volatile bool isRunning = false;

        // Game state tracking
        private Dictionary<string, object> gameState = new Dictionary<string, object>();
        private Dictionary<string, float> metrics = new Dictionary<string, float>();
        private object stateLock = new object();

        // Mod information
        public string Name => "RL Agent Bridge";
        public string Description => "Bridges Cities: Skylines with an external reinforcement learning agent";

        // Called when the mod is enabled
        public void OnEnabled()
        {
            Debug.Log("RL Agent Bridge: Mod enabled");
        }

        // Called when the mod is disabled
        public void OnDisabled()
        {
            StopServer();
            Debug.Log("RL Agent Bridge: Mod disabled");
        }

        #region ILoadingExtension Implementation

        public void OnCreated(ILoading loading)
        {
            Debug.Log("RL Agent Bridge: Loading extension created");
        }

        public void OnReleased()
        {
            StopServer();
            Debug.Log("RL Agent Bridge: Loading extension released");
        }

        public void OnLevelLoaded(LoadMode mode)
        {
            // Only start the server when a game is loaded
            if (mode == LoadMode.LoadGame || mode == LoadMode.NewGame)
            {
                StartServer();
                Debug.Log($"RL Agent Bridge: Server started on http://{ApiHost}:{ApiPort}/");
            }
        }

        public void OnLevelUnloading()
        {
            StopServer();
            Debug.Log("RL Agent Bridge: Level unloading, server stopped");
        }

        #endregion

        #region IThreadingExtension Implementation

        public void OnCreated(IThreading threading)
        {
            Debug.Log("RL Agent Bridge: Threading extension created");
        }

        public void OnUpdate(float realTimeDelta, float simulationTimeDelta)
        {
            // This runs on the main thread, update game state here
            if (isRunning)
            {
                UpdateGameState();
            }
        }

        public void OnBeforeSimulationTick()
        {
            // Called before simulation tick
        }

        public void OnBeforeSimulationFrame()
        {
            // Called before simulation frame
        }

        public void OnAfterSimulationFrame()
        {
            // Called after simulation frame
        }

        public void OnAfterSimulationTick()
        {
            // Called after simulation tick
        }

        #endregion

        #region HTTP Server Implementation

        private void StartServer()
        {
            if (isRunning) return;

            try
            {
                listener = new HttpListener();
                listener.Prefixes.Add($"http://{ApiHost}:{ApiPort}/");
                listener.Start();

                isRunning = true;
                listenerThread = new Thread(HandleRequests);
                listenerThread.IsBackground = true;
                listenerThread.Start();

                Debug.Log("RL Agent Bridge: HTTP server started");
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Failed to start server: {ex.Message}");
            }
        }

        private void StopServer()
        {
            if (!isRunning) return;

            try
            {
                isRunning = false;
                
                if (listener != null)
                {
                    listener.Stop();
                    listener.Close();
                    listener = null;
                }

                if (listenerThread != null && listenerThread.IsAlive)
                {
                    listenerThread.Join(1000); // Wait for thread to finish
                    if (listenerThread.IsAlive)
                    {
                        listenerThread.Abort();
                    }
                    listenerThread = null;
                }

                Debug.Log("RL Agent Bridge: HTTP server stopped");
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error stopping server: {ex.Message}");
            }
        }

        private void HandleRequests()
        {
            while (isRunning)
            {
                try
                {
                    var context = listener.GetContext();
                    ThreadPool.QueueUserWorkItem((_) => ProcessRequest(context));
                }
                catch (Exception ex)
                {
                    if (isRunning)
                    {
                        Debug.LogError($"RL Agent Bridge: Error handling request: {ex.Message}");
                    }
                }
            }
        }

        private void ProcessRequest(HttpListenerContext context)
        {
            try
            {
                var request = context.Request;
                var response = context.Response;

                // Set CORS headers to allow requests from any origin
                response.Headers.Add("Access-Control-Allow-Origin", "*");
                response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                response.Headers.Add("Access-Control-Allow-Headers", "Content-Type");

                // Handle OPTIONS requests (CORS preflight)
                if (request.HttpMethod == "OPTIONS")
                {
                    response.StatusCode = 200;
                    response.Close();
                    return;
                }

                string responseText = "";

                // Route the request based on the URL
                if (request.Url.AbsolutePath == "/state")
                {
                    // Return the current game state
                    responseText = GetGameStateJson();
                    response.ContentType = "application/json";
                }
                else if (request.Url.AbsolutePath == "/action" && request.HttpMethod == "POST")
                {
                    // Process an action
                    using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
                    {
                        string requestBody = reader.ReadToEnd();
                        bool success = ProcessAction(requestBody);
                        responseText = $"{{\"success\": {success.ToString().ToLower()}}}";
                    }
                    response.ContentType = "application/json";
                }
                else
                {
                    // Unknown endpoint
                    response.StatusCode = 404;
                    responseText = "Not Found";
                    response.ContentType = "text/plain";
                }

                // Send the response
                byte[] buffer = Encoding.UTF8.GetBytes(responseText);
                response.ContentLength64 = buffer.Length;
                response.OutputStream.Write(buffer, 0, buffer.Length);
                response.Close();
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error processing request: {ex.Message}");
                try
                {
                    context.Response.StatusCode = 500;
                    context.Response.Close();
                }
                catch { /* Ignore errors when closing the response */ }
            }
        }

        #endregion

        #region Game State and Actions

        private void UpdateGameState()
        {
            try
            {
                lock (stateLock)
                {
                    // Update basic game info
                    gameState["timestamp"] = DateTime.Now.ToString("o");
                    gameState["simulationPaused"] = SimulationManager.instance.SimulationPaused;
                    gameState["simulationSpeed"] = SimulationManager.instance.SelectedSimulationSpeed;
                    
                    // Update metrics
                    UpdateMetrics();
                    gameState["metrics"] = metrics;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error updating game state: {ex.Message}");
            }
        }

        private void UpdateMetrics()
        {
            try
            {
                // Population
                metrics["population"] = Singleton<StatisticsManager>.instance.Residents.m_finalCount;
                
                // Happiness (average)
                metrics["happiness"] = Singleton<CitizenManager>.instance.GetAverageHappiness();
                
                // Budget
                metrics["budget_balance"] = Singleton<EconomyManager>.instance.CurrentCashAmount / 100f;
                
                // Traffic flow
                metrics["traffic_flow"] = Singleton<TrafficManager>.instance.m_trafficFlow;
                
                // Pollution (average)
                metrics["pollution"] = Singleton<DistrictManager>.instance.m_districts.m_buffer[0].m_pollutionRate;
                
                // Land value (average)
                metrics["land_value"] = Singleton<DistrictManager>.instance.m_districts.m_buffer[0].m_landValue;
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error updating metrics: {ex.Message}");
            }
        }

        private string GetGameStateJson()
        {
            lock (stateLock)
            {
                // Simple JSON serialization
                var sb = new StringBuilder();
                sb.Append("{");
                
                // Add basic game info
                sb.Append($"\"timestamp\": \"{gameState["timestamp"]}\",");
                sb.Append($"\"simulationPaused\": {gameState["simulationPaused"].ToString().ToLower()},");
                sb.Append($"\"simulationSpeed\": {gameState["simulationSpeed"]},");
                
                // Add metrics
                sb.Append("\"metrics\": {");
                int metricCount = 0;
                foreach (var metric in metrics)
                {
                    sb.Append($"\"{metric.Key}\": {metric.Value}");
                    if (++metricCount < metrics.Count)
                    {
                        sb.Append(",");
                    }
                }
                sb.Append("}");
                
                sb.Append("}");
                return sb.ToString();
            }
        }

        private bool ProcessAction(string actionJson)
        {
            try
            {
                // For now, just log the action
                Debug.Log($"RL Agent Bridge: Received action: {actionJson}");
                
                // TODO: Parse the action JSON and perform the corresponding action in the game
                // This will require dispatching to the main thread for most game operations
                
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error processing action: {ex.Message}");
                return false;
            }
        }

        #endregion
    }
} 