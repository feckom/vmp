Key Features of the Visual Music Player (VMP)
Real-time Audio Visualization: Creates dynamic, reactive visuals synchronized to the audio spectrum (bass, mids, highs) and rhythmic beats of the currently playing music.
Central "Flubber" Organism: Features a complex, pulsating, amorphous central shape (the Flubber) that reacts to the music's intensity, particularly the bass. Its size, shape, and texture morph organically.
3D Lighting & Depth Effects: Applies simulated 3D lighting, depth, and specular highlights to the Flubber, giving it a more volumetric and realistic appearance that reacts to the music.
Rotating Energy Beams: Displays animated beams or rays radiating from the center, whose length and thickness pulse with the music's energy.
Beat Detection & Response: Detects musical beats and triggers visual events like shockwaves (ripples), satellite spawns, and body ripples on the Flubber.
Shape Morphing: The Flubber periodically and automatically morphs between predefined geometric and organic shapes (e.g., Circle, Star, Heart, Blob, Hexagon) based on the song's timeline.
Letter/Word Morphing: Can display and morph individual letters or words (from a predefined "Cyberpunk" lexicon) through the Flubber's shape.
Lyrics Display (LRC): Can load and display synchronized lyrics (from .lrc files) at the top of the screen when enabled.
Customizable Backgrounds: Supports loading and cycling through user-provided background images.
Interactive UI: Offers a comprehensive user interface for controlling playback (play/pause, next/previous track, seek), volume, shuffle, repeat modes (Off, All, One), fullscreen toggle, and visualization presets.
Performance Information: Displays real-time performance stats like FPS and estimated BPM (Beats Per Minute).
Track Information: Shows the current track's title and artist, along with playback time.
Visual Presets: Allows cycling through different combinations of UI elements (HUD, FPS, Time, Title) for a customized viewing experience.
Low-CPU Mode: Includes an option for less computationally intensive visualizations.
Sepia Filter: Can apply a sepia tone filter to the entire screen for a different aesthetic.
Program Behavior
The program behaves as a highly reactive, music-driven visualizer:

Initialization: Scans a music directory for audio files, loads configuration, and initializes audio and graphics systems. It preloads the first track.
Playback & Analysis: Plays the selected audio track. In a background thread, it continuously analyzes the audio stream to extract frequency band data (for the beams), bass energy (for Flubber size and beat detection), and voice energy (for the inner voice circle).
Visual Update Loop: On each frame, it:
Updates the Flubber's position with subtle physics (it can "jiggle" on strong beats).
Calculates the Flubber's current shape (either reactive, a static letter, or a morphing geometric shape).
Computes the 3D lighting and depth effects for the Flubber.
Draws the rotating energy beams, with yellow "peak" highlights for emphasis.
Renders the Flubber ring with its calculated fill and edge colors, applying the 3D effects.
Draws any active satellites or ripple effects triggered by detected beats.
Displays the voice energy circle.
Renders a progress arc around the Flubber.
Overlays the UI text (track info, time, HUD, FPS, etc.) and any active lyrics.
Event-Driven: Responds to user input (keyboard, mouse) for navigation, control, and UI toggling. It also responds to audio events (beats) by spawning visual effects.
State Management: Manages playback state (play, pause), track queue, shuffle, repeat modes, and remembers the last used volume and visualization preset.
Resource Management: Uses caching for audio analysis and glyph (letter) profiles to improve performance. It prefetches the next track in the background.
Random Events Displayed
The program generates the following random visual events:

Satellite Spawning: When a beat is detected, 1 to 3 small, orbiting "satellite" circles are randomly spawned around the Flubber. Their initial angle, distance, size, and orbit speed are randomized.
Shape Morphing Timing: The timing for when the Flubber initiates a morph to a new geometric shape is planned semi-randomly within the song's duration (after an initial delay and with minimum gaps).
Shape Selection: The specific geometric shape (Circle, Star, Heart, etc.) the Flubber morphs into is chosen randomly from a curated list, with logic to avoid immediately repeating visually similar shapes.
Letter/Word Selection & Timing: The words displayed via letter morphing are chosen randomly from the CyberpunkLexicon (e.g., "NEON", "HACK", "CORE"). The start times for these word sequences within the song are also randomized.
Background Image Selection: When backgrounds are enabled, the initial background and the one shown when cycling with [ or ] are chosen randomly from the available images in the backgrounds folder.
Organic Texture: The Flubber's surface has a subtle, randomized noise texture overlay.
Physics Jitter: The Flubber's center position receives small, random jitter impulses based on the bass energy, making its movement less mechanical.
Morph Duration & Dwell Time: The duration of shape morphs and the time spent dwelling on a static letter or shape are randomized within defined ranges.
Ripple Effects: The body ripple effect on the Flubber, triggered by beats, has randomized phase and animation.
