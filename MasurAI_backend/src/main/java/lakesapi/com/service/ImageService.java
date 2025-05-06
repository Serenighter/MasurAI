package lakesapi.com.service;

import lakesapi.com.common.PythonRunner;
import lakesapi.com.exception.ImageNotFoundException;
import lakesapi.com.exception.ScriptExecutionException;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;

@Service
public class ImageService {
  private static final String IMAGE_DIRECTORY = "src/main/resources/analyze/SatellitePics/";

  public byte[] getImage(
      @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
      @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to) {
    if (!(imageExists(from))) {
      throw new ImageNotFoundException("Photo with date " + from.toString() + "is not found");
    }

    if (!(imageExists(to))) {
      throw new ImageNotFoundException("Photo with date " + to.toString() + "is not found");
    }

    changeImagePaths(from.toString() + ".jpg", to.toString() + ".jpg");

    try {
      PythonRunner runner = new PythonRunner("src/main/resources/analyze/.venv/Scripts/python.exe");
      runner.start("src/main/resources/analyze/image_analyze.py");
      int code = runner.waitForFinish();
      System.out.println("Exit code = " + code);

    } catch (Exception e) {
      throw new ScriptExecutionException(e.getMessage());
    }

    return exportPhoto("src/main/resources/analyze/analyzedChart.png");
  }

  /**
   * Sprawdza czy obraz o podanej dacie istnieje
   *
   * @param date data obrazu (bez rozszerzenia .jpg)
   * @return true jeśli obraz istnieje, false w przeciwnym przypadku
   */
  private boolean imageExists(@DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date) {
    Resource resource = new ClassPathResource("analyze/SatellitePics/" + date.toString() + ".jpg");
    boolean exists = resource.exists();
    System.out.println("Checking image: " + date.toString() + ".jpg - exists: " + exists);
    return exists;
  }

  private boolean changeImagePaths(String newFirstImagePath, String newSecondImagePath) {
    String folderPath = "src/main/resources/analyze/";
    try {
      // Zmiana zawartości pliku firstImagePath.txt
      Files.write(Paths.get(folderPath + "firstImagePath.txt"), newFirstImagePath.getBytes());

      // Zmiana zawartości pliku secondImagePath.txt
      Files.write(Paths.get(folderPath + "secondImagePath.txt"), newSecondImagePath.getBytes());

      return true; // Wszystko poszło ok
    } catch (IOException e) {
      e.printStackTrace();
      return false; // Wystąpił błąd przy zapisywaniu
    }
  }

  private byte[] exportPhoto(String pathName) {
    try {
      // Ścieżka do pliku obrazu
      Path imagePath = Path.of(pathName); // Używamy pełnej ścieżki z parametru pathName

      // Wczytanie obrazu do tablicy bajtów
      return Files.readAllBytes(imagePath);

    } catch (IOException e) {
      // Obsługa błędów, gdy nie uda się wczytać pliku
      System.err.println("Błąd wczytywania obrazu: " + pathName);
      e.printStackTrace();
      return null; // Zwróć null, jeśli wystąpi błąd
    }
  }
}
