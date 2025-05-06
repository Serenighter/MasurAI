package lakesapi.com.infrastructure;

import lakesapi.com.exception.ImageNotFoundException;
import lakesapi.com.service.ImageService;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDate;

@RestController
@RequestMapping("/api/image")
class ImageController {
  private final ImageService imageService;

  public ImageController(ImageService imageService) {
    this.imageService = imageService;
  }

  @GetMapping("date/{from}/{to}")
  public ResponseEntity<byte[]> getImage(
      @PathVariable @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
      @PathVariable @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to) {

    try {
      byte[] image = imageService.getImage(from, to);

      // Dodanie nagłówków HTTP
      HttpHeaders headers = new HttpHeaders();
      headers.setContentType(MediaType.IMAGE_PNG); // Ustaw typ na PNG
      headers.setContentLength(image.length); // Ustaw długość odpowiedzi

      return new ResponseEntity<>(image, headers, HttpStatus.OK);

    } catch (ImageNotFoundException e) {
      return ResponseEntity.status(HttpStatus.NOT_FOUND).body(new byte[0]);
    }
  }
}
