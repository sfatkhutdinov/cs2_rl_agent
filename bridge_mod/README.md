# Cities: Skylines 2 RL Agent Bridge Mod

This mod serves as a bridge between Cities: Skylines and the external reinforcement learning agent. It exposes a simple REST API that allows the Python agent to interact with the game.

## Installation

1. Locate your Cities: Skylines mods folder:
   - Windows: `C:\Users\<username>\AppData\Local\Colossal Order\Cities_Skylines\Addons\Mods`
   - Mac: `/Users/<username>/Library/Application Support/Colossal Order/Cities_Skylines/Addons/Mods`
   - Linux: `/home/<username>/.local/share/Colossal Order/Cities_Skylines/Addons/Mods/`

2. Create a new folder named `RLAgentBridge` in the mods directory.

3. Copy the contents of this `bridge_mod` folder to the `RLAgentBridge` folder.

4. Start Cities: Skylines and enable the "RL Agent Bridge" mod in the Content Manager.

## How It Works

The bridge mod starts a local HTTP server on port 5000 when a game is loaded. The Python agent can then communicate with the game through this server using the following endpoints:

- `GET http://localhost:5000/state` - Returns the current game state as JSON
- `POST http://localhost:5000/action` - Performs an action in the game

## API Reference

### Game State

The game state JSON includes:

```json
{
  "timestamp": "2023-03-10T12:34:56.789Z",
  "simulationPaused": false,
  "simulationSpeed": 1,
  "metrics": {
    "population": 10000,
    "happiness": 85.5,
    "budget_balance": 50000.0,
    "traffic_flow": 92.3,
    "pollution": 15.2,
    "land_value": 45.7
  }
}
```

### Actions

Actions are sent as JSON in the request body:

```json
{
  "type": "zone",
  "zone_type": "residential",
  "position": {
    "x": 100.0,
    "y": 0.0,
    "z": 200.0
  }
}
```

Supported action types:
- `zone` - Create a zone (residential, commercial, industrial, office)
- `infrastructure` - Build infrastructure (roads, power, water)
- `budget` - Adjust budget settings

## Troubleshooting

- If the mod doesn't appear in the Content Manager, make sure the files are in the correct location.
- Check the game's output log for any error messages from the mod.
- Make sure no other application is using port 5000 on your system.

## Development

This mod is part of the CS2 RL Agent project. For more information, see the main project repository. 