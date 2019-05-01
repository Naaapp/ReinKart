--[[
Play.lua
This script is used to actually play the Convolutional AI on a track. It asynchronously communicates
with predict-server.py over a TCP socket. The message protocol is a simple, line-oriented feed.
This module can also be called as a function, in which case the first argument is the number of
frames to play for.
]]--
TRACKS = {'states/TT/LR.state', 'states/GP/MMF.state', 'states/TT/CM.state'}

--[[ BEGIN CONFIGURATION ]]--
USE_CLIPBOARD = true -- Use the clipboard to send screenshots to the predict server.

--[[ How many frames to wait before sending a new prediction request. If you're using a file, you
may want to consider adding some frames here. ]]--
WAIT_FRAMES = 10

USE_MAPPING = true -- Whether or not to use input remapping.
CHECK_PROGRESS_EVERY = 40 -- Check progress after this many frames to detect if we get stuck.
CHECK_PREVIOUS_SCORE_EVERY = 4
--[[ END CONFIGURATION ]]--

local chunk_args = {...}
local PLAY_FOR_FRAMES = chunk_args[1]
if PLAY_FOR_FRAMES ~= nil then print("Playing for " .. PLAY_FOR_FRAMES .. " frames.") end

local util = require("util")

local SCREENSHOT_FILE = util.getTMPDir() .. '\\predict-screenshot.png'

local tcp = require("lualibs.socket").tcp()
local success, error = tcp:connect('localhost', 36296)
if not success then
  print("Failed to connect to server:", error)
  return
end

client.setscreenshotosd(false)

--[[local course = util.readCourse()
tcp:send("COURSE:" .. course .. "\n")
--]]


client.unpause()

local max_progress = util.readProgress()
local score = 1
local previous_progress = 0
local current_progress  = 0
local frame = 1
local init = 0



outgoing_message, outgoing_message_index = nil, nil
function request_prediction()

  --actuellement la progression max est +/- 0.01 (ça dépend de la vitesse aussi), vitesse max +/- 6

  local dif = current_progress - previous_progress
  local velocity = util.readVelocity()
  local cos = util.readPlayerCos()
  local sin = util.readPlayerSin()
  local vx = util.readPlayerXV()
  local vy = util.readPlayerYV()
  local distance = current_progress


  --score = ((dif*80)*(dif*80)*(1/velocity))*10 + 1
  score = ((dif*170)*(dif*170))*(velocity/4) + 1

  


  if score > 2 then
    score = 2
  end
  if dif <= 0 then
    score = 0
  end
  --print(score)
  --print(dif)
  
  --print(score)

  if frame == 1 then
    init = 1
  else
    init = 0
  end



  if USE_CLIPBOARD then
    outgoing_message = "MESSAGE"
                      ..tostring(init)
                      .."SCORE"
                      ..string.format("%.14f", score)
                      .."DISTANCE"
                      ..string.format("%.14f", distance)
                      .."COS"
                      ..string.format("%.14f", cos)
                      .."SIN"
                      ..string.format("%.14f", sin)
                      .."VELOCITY"
                      ..string.format("%.14f", velocity)
                      .."VX"
                      ..string.format("%.14f", vx)
                      .."VY"
                      ..string.format("%.14f", vy)
                      .."\n"
  else
    client.screenshot(SCREENSHOT_FILE)
    outgoing_message = "PREDICT:"..tostring(init)..tostring(score).. SCREENSHOT_FILE .. "\n"
  end
  outgoing_message_index = 1
end
request_prediction()

local receive_buffer = ""

function onexit()
  client.pause()
  tcp:close()
end
local exit_guid = event.onexit(onexit)

local current_action = 0

local esc_prev = input.get()['Escape']

BOX_CENTER_X, BOX_CENTER_Y = 160, 215
BOX_WIDTH, BOX_HEIGHT = 100, 4
SLIDER_WIDTH, SLIDER_HIEGHT = 4, 16
function draw_info()
  gui.drawBox(BOX_CENTER_X - BOX_WIDTH / 2, BOX_CENTER_Y - BOX_HEIGHT / 2,
              BOX_CENTER_X + BOX_WIDTH / 2, BOX_CENTER_Y + BOX_HEIGHT / 2,
              none, 0x60FFFFFF)
  gui.drawBox(BOX_CENTER_X + current_action*(BOX_WIDTH / 2) - SLIDER_WIDTH / 2, BOX_CENTER_Y - SLIDER_HIEGHT / 2,
              BOX_CENTER_X + current_action*(BOX_WIDTH / 2) + SLIDER_WIDTH / 2, BOX_CENTER_Y + SLIDER_HIEGHT / 2,
              none, 0xFFFF0000)
end



while util.readProgress() < 3 do



  while frame < 10 do
  joypad.set({["P1 A"] = true})
  joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(0) })
  emu.frameadvance()
  frame = frame + 1 
  end

  -- Process the outgoing message.
  if outgoing_message ~= nil then
    local sent, error, last_byte = tcp:send(outgoing_message, outgoing_message_index)
    if sent ~= nil then
      outgoing_message = nil
      outgoing_message_index = nil
    else
      if error == "timeout" then
        outgoing_message_index = last_byte + 1
      else
        print("Send failed: ", error); break
      end
    end
  end


  local message, error
  message, error, receive_buffer = tcp:receive("*l", receive_buffer)
  --message [0] : straight, message [0] : left, message [0] : right
  if message == nil then
    if error ~= "timeout" then
      print("Receive failed: ", error); 
    end
  else
    if message ~= "PREDICTIONERROR" then



      if tonumber(string.sub(message, 1, 1)) == 1 then
        current_action = 0
      end
      if tonumber(string.sub(message, 2, 2)) == 1 then
        current_action =  -1
      end
      if tonumber(string.sub(message, 3, 3)) == 1 then
        current_action =  1
      end
      --print(current_action)


      --[[print(score)--]]

      previous_progress = current_progress

      for i=1, WAIT_FRAMES do
        joypad.set({["P1 A"] = true})
        joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(current_action) })
        draw_info()
        emu.frameadvance()
      end
    else
      print("Prediction error...")
    end
    current_progress = util.readProgress()

    request_prediction()
  end



  joypad.set({["P1 A"] = true})
  joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(current_action) })
  draw_info()
  emu.frameadvance()

--[[  if PLAY_FOR_FRAMES ~= nil then
    if PLAY_FOR_FRAMES > 0 then PLAY_FOR_FRAMES = PLAY_FOR_FRAMES - 1
    elseif PLAY_FOR_FRAMES == 0 then break end
  end--]]

  frame = frame + 1

  current_progress = util.readProgress()

  -- if we haven't made any progress since the last check, just break.
  if frame > 50 then
    if frame % (CHECK_PROGRESS_EVERY / 10) == 0 then
      if current_progress < previous_progress then
        print(current_progress)
        break
      end
    end
    if frame % CHECK_PROGRESS_EVERY == 0 then
      if current_progress == previous_progress then
        print(current_progress)
        break
      else max_progress = current_progress end
    end
  end

  if not esc_prev and input.get()['Escape'] then break end
  esc_prev = input.get()['Escape']
end

-- print("exit")

onexit()
event.unregisterbyid(exit_guid)