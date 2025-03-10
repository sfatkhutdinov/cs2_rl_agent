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
                ThreadingExtension.dispatcher.ProcessQueue();
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
                Debug.Log($"RL Agent Bridge: Received action: {actionJson}");
                
                // Deserialize the action JSON
                Dictionary<string, object> action = ParseJson(actionJson);
                if (action == null || !action.ContainsKey("type"))
                {
                    Debug.LogError("RL Agent Bridge: Invalid action format - missing 'type' field");
                    return false;
                }
                
                string actionType = action["type"].ToString();
                
                // Dispatch the action based on its type
                switch (actionType)
                {
                    case "zone":
                        return HandleZoneAction(action);
                    
                    case "infrastructure":
                        return HandleInfrastructureAction(action);
                    
                    case "budget":
                        return HandleBudgetAction(action);
                    
                    case "game_control":
                        return HandleGameControlAction(action);
                    
                    default:
                        Debug.LogError($"RL Agent Bridge: Unknown action type: {actionType}");
                        return false;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error processing action: {ex.Message}");
                return false;
            }
        }
        
        private Dictionary<string, object> ParseJson(string json)
        {
            try
            {
                // Simple JSON parser for the action format we expect
                Dictionary<string, object> result = new Dictionary<string, object>();
                
                // Remove curly braces and whitespace
                json = json.Trim();
                if (json.StartsWith("{")) json = json.Substring(1);
                if (json.EndsWith("}")) json = json.Substring(0, json.Length - 1);
                
                // Split by commas, but not those inside nested objects
                List<string> parts = new List<string>();
                int depth = 0;
                int startIndex = 0;
                
                for (int i = 0; i < json.Length; i++)
                {
                    char c = json[i];
                    if (c == '{') depth++;
                    else if (c == '}') depth--;
                    else if (c == ',' && depth == 0)
                    {
                        parts.Add(json.Substring(startIndex, i - startIndex));
                        startIndex = i + 1;
                    }
                }
                
                // Add the last part
                if (startIndex < json.Length)
                    parts.Add(json.Substring(startIndex));
                
                // Parse each key-value pair
                foreach (string part in parts)
                {
                    string[] keyValue = part.Split(new[] { ':' }, 2);
                    if (keyValue.Length == 2)
                    {
                        string key = keyValue[0].Trim().Trim('"');
                        string value = keyValue[1].Trim();
                        
                        // Parse the value based on its format
                        if (value.StartsWith("\"") && value.EndsWith("\""))
                        {
                            // String value
                            result[key] = value.Substring(1, value.Length - 2);
                        }
                        else if (value.StartsWith("{") && value.EndsWith("}"))
                        {
                            // Nested object
                            result[key] = ParseJson(value);
                        }
                        else if (value == "true")
                        {
                            result[key] = true;
                        }
                        else if (value == "false")
                        {
                            result[key] = false;
                        }
                        else if (float.TryParse(value, out float floatValue))
                        {
                            result[key] = floatValue;
                        }
                        else
                        {
                            // Default to string if parsing fails
                            result[key] = value;
                        }
                    }
                }
                
                return result;
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error parsing JSON: {ex.Message}");
                return null;
            }
        }
        
        private bool HandleZoneAction(Dictionary<string, object> action)
        {
            try
            {
                if (!action.ContainsKey("zone_type") || !action.ContainsKey("position"))
                {
                    Debug.LogError("RL Agent Bridge: Zone action missing required fields");
                    return false;
                }
                
                string zoneType = action["zone_type"].ToString();
                Dictionary<string, object> position = action["position"] as Dictionary<string, object>;
                
                if (position == null || !position.ContainsKey("x") || !position.ContainsKey("z"))
                {
                    Debug.LogError("RL Agent Bridge: Invalid position format");
                    return false;
                }
                
                float x = Convert.ToSingle(position["x"]);
                float z = Convert.ToSingle(position["z"]);
                
                // Queue the action to be executed on the main thread
                ThreadingExtension.dispatcher.QueueMainThread(() =>
                {
                    // Convert zone type to game's zone type
                    ItemClass.Service service = ItemClass.Service.None;
                    ItemClass.SubService subService = ItemClass.SubService.None;
                    
                    switch (zoneType)
                    {
                        case "residential":
                            service = ItemClass.Service.Residential;
                            subService = ItemClass.SubService.ResidentialLow;
                            break;
                        case "commercial":
                            service = ItemClass.Service.Commercial;
                            subService = ItemClass.SubService.CommercialLow;
                            break;
                        case "industrial":
                            service = ItemClass.Service.Industrial;
                            subService = ItemClass.SubService.IndustrialGeneric;
                            break;
                        case "office":
                            service = ItemClass.Service.Office;
                            subService = ItemClass.SubService.OfficeGeneric;
                            break;
                        default:
                            Debug.LogError($"RL Agent Bridge: Unknown zone type: {zoneType}");
                            return;
                    }
                    
                    // Create a zone at the specified position
                    Vector3 worldPos = new Vector3(x, 0, z);
                    ZoneManager.instance.CreateZone(service, subService, worldPos, 40f);
                    
                    Debug.Log($"RL Agent Bridge: Created {zoneType} zone at ({x}, {z})");
                });
                
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error handling zone action: {ex.Message}");
                return false;
            }
        }
        
        private bool HandleInfrastructureAction(Dictionary<string, object> action)
        {
            try
            {
                if (!action.ContainsKey("infra_type") || !action.ContainsKey("position"))
                {
                    Debug.LogError("RL Agent Bridge: Infrastructure action missing required fields");
                    return false;
                }
                
                string infraType = action["infra_type"].ToString();
                Dictionary<string, object> position = action["position"] as Dictionary<string, object>;
                
                if (position == null || !position.ContainsKey("x") || !position.ContainsKey("z"))
                {
                    Debug.LogError("RL Agent Bridge: Invalid position format");
                    return false;
                }
                
                float x = Convert.ToSingle(position["x"]);
                float z = Convert.ToSingle(position["z"]);
                
                // Optional end position for roads, power lines, etc.
                float endX = x;
                float endZ = z;
                
                if (action.ContainsKey("end_position"))
                {
                    Dictionary<string, object> endPosition = action["end_position"] as Dictionary<string, object>;
                    if (endPosition != null && endPosition.ContainsKey("x") && endPosition.ContainsKey("z"))
                    {
                        endX = Convert.ToSingle(endPosition["x"]);
                        endZ = Convert.ToSingle(endPosition["z"]);
                    }
                }
                
                // Queue the action to be executed on the main thread
                ThreadingExtension.dispatcher.QueueMainThread(() =>
                {
                    Vector3 startPos = new Vector3(x, 0, z);
                    Vector3 endPos = new Vector3(endX, 0, endZ);
                    
                    switch (infraType)
                    {
                        case "road_straight":
                            // Find a basic road prefab
                            NetInfo roadPrefab = FindPrefab<NetInfo>("Basic Road");
                            if (roadPrefab != null)
                            {
                                NetManager.instance.CreateSegment(roadPrefab, startPos, endPos, 0, 0);
                                Debug.Log($"RL Agent Bridge: Created road from ({x}, {z}) to ({endX}, {endZ})");
                            }
                            else
                            {
                                Debug.LogError("RL Agent Bridge: Road prefab not found");
                            }
                            break;
                            
                        case "power":
                            // Find power line prefab
                            NetInfo powerPrefab = FindPrefab<NetInfo>("Electricity Pole");
                            if (powerPrefab != null)
                            {
                                NetManager.instance.CreateSegment(powerPrefab, startPos, endPos, 0, 0);
                                Debug.Log($"RL Agent Bridge: Created power line from ({x}, {z}) to ({endX}, {endZ})");
                            }
                            else
                            {
                                Debug.LogError("RL Agent Bridge: Power prefab not found");
                            }
                            break;
                            
                        case "water":
                            // Find water pipe prefab
                            NetInfo waterPrefab = FindPrefab<NetInfo>("Water Pipe");
                            if (waterPrefab != null)
                            {
                                NetManager.instance.CreateSegment(waterPrefab, startPos, endPos, 0, 0);
                                Debug.Log($"RL Agent Bridge: Created water pipe from ({x}, {z}) to ({endX}, {endZ})");
                            }
                            else
                            {
                                Debug.LogError("RL Agent Bridge: Water pipe prefab not found");
                            }
                            break;
                            
                        default:
                            Debug.LogError($"RL Agent Bridge: Unknown infrastructure type: {infraType}");
                            break;
                    }
                });
                
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error handling infrastructure action: {ex.Message}");
                return false;
            }
        }
        
        private bool HandleBudgetAction(Dictionary<string, object> action)
        {
            try
            {
                if (!action.ContainsKey("budget_action"))
                {
                    Debug.LogError("RL Agent Bridge: Budget action missing required fields");
                    return false;
                }
                
                string budgetAction = action["budget_action"].ToString();
                
                // Queue the action to be executed on the main thread
                ThreadingExtension.dispatcher.QueueMainThread(() =>
                {
                    // Parse the budget action (e.g., "increase_residential_budget")
                    string[] parts = budgetAction.Split('_');
                    if (parts.Length < 3)
                    {
                        Debug.LogError($"RL Agent Bridge: Invalid budget action format: {budgetAction}");
                        return;
                    }
                    
                    string direction = parts[0]; // "increase" or "decrease"
                    string serviceStr = parts[1]; // "residential", "commercial", etc.
                    
                    // Map to game's service type
                    ItemClass.Service service = ItemClass.Service.None;
                    switch (serviceStr)
                    {
                        case "residential":
                            service = ItemClass.Service.Residential;
                            break;
                        case "commercial":
                            service = ItemClass.Service.Commercial;
                            break;
                        case "industrial":
                            service = ItemClass.Service.Industrial;
                            break;
                        case "transport":
                            service = ItemClass.Service.PublicTransport;
                            break;
                        default:
                            Debug.LogError($"RL Agent Bridge: Unknown service type: {serviceStr}");
                            return;
                    }
                    
                    // Get current budget
                    int currentBudget = EconomyManager.instance.GetBudget(service);
                    
                    // Apply change
                    int newBudget = currentBudget;
                    if (direction == "increase")
                    {
                        newBudget = Math.Min(currentBudget + 10, 150); // Increase by 10%, max 150%
                    }
                    else if (direction == "decrease")
                    {
                        newBudget = Math.Max(currentBudget - 10, 50); // Decrease by 10%, min 50%
                    }
                    
                    // Set new budget
                    EconomyManager.instance.SetBudget(service, newBudget);
                    Debug.Log($"RL Agent Bridge: {direction}d {serviceStr} budget from {currentBudget}% to {newBudget}%");
                });
                
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error handling budget action: {ex.Message}");
                return false;
            }
        }
        
        private bool HandleGameControlAction(Dictionary<string, object> action)
        {
            try
            {
                if (!action.ContainsKey("control_type"))
                {
                    Debug.LogError("RL Agent Bridge: Game control action missing required fields");
                    return false;
                }
                
                string controlType = action["control_type"].ToString();
                
                // Queue the action to be executed on the main thread
                ThreadingExtension.dispatcher.QueueMainThread(() =>
                {
                    switch (controlType)
                    {
                        case "speed":
                            if (action.ContainsKey("value"))
                            {
                                int speed = Convert.ToInt32(action["value"]);
                                if (speed >= 1 && speed <= 3)
                                {
                                    // Set game speed
                                    SimulationManager.instance.SelectedSimulationSpeed = speed;
                                    Debug.Log($"RL Agent Bridge: Set game speed to {speed}");
                                }
                                else
                                {
                                    Debug.LogError($"RL Agent Bridge: Invalid game speed: {speed}");
                                }
                            }
                            break;
                            
                        case "pause":
                            // Toggle pause state
                            SimulationManager.instance.SimulationPaused = !SimulationManager.instance.SimulationPaused;
                            Debug.Log($"RL Agent Bridge: Set pause state to {SimulationManager.instance.SimulationPaused}");
                            break;
                            
                        case "reset":
                            // Reset game not directly supported - would need to load a saved game
                            Debug.LogError("RL Agent Bridge: Reset game not directly supported");
                            break;
                            
                        default:
                            Debug.LogError($"RL Agent Bridge: Unknown game control type: {controlType}");
                            break;
                    }
                });
                
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"RL Agent Bridge: Error handling game control action: {ex.Message}");
                return false;
            }
        }
        
        private T FindPrefab<T>(string name) where T : PrefabInfo
        {
            int count = PrefabCollection<T>.LoadedCount();
            for (uint i = 0; i < count; i++)
            {
                T prefab = PrefabCollection<T>.GetLoaded(i);
                if (prefab != null && prefab.name.Contains(name))
                {
                    return prefab;
                }
            }
            return null;
        }

        // Add this class to handle dispatching to the main thread
        public static class ThreadingExtension
        {
            public static QueuedDispatcher dispatcher = new QueuedDispatcher();
            
            public class QueuedDispatcher
            {
                private readonly Queue<Action> queue = new Queue<Action>();
                
                public void QueueMainThread(Action action)
                {
                    lock (queue)
                    {
                        queue.Enqueue(action);
                    }
                }
                
                public void ProcessQueue()
                {
                    lock (queue)
                    {
                        while (queue.Count > 0)
                        {
                            Action action = queue.Dequeue();
                            try
                            {
                                action();
                            }
                            catch (Exception ex)
                            {
                                Debug.LogError($"RL Agent Bridge: Error processing queued action: {ex.Message}");
                            }
                        }
                    }
                }
            }
        }
    }
} 